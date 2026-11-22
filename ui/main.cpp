#include <QApplication>
#include <QCommandLineParser>
#include <QKeyEvent>
#include <QLabel>
#include <QMouseEvent>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>
#include <cmath>

#include "dibrwidget.h"
#include "eyelink.h"

/**
 * 2.5D 怀念影像 —— 查看器界面
 *
 * 交互：
 *   鼠标移动   改变视点（模拟头部左右/上下移动）
 *   空格       自动摆动（无人操作时演示用）
 *   H          显示/隐藏 HUD
 *   F          全屏
 *   + / -      视差强度
 *   [ / ]      焦点深度
 *   Q / Esc    退出
 *
 * 离屏验证：
 *   ./viewer --asset ../assets/demo --snapshot out.png --eye 0.6 0
 */

class Viewer : public QWidget
{
public:
    Viewer(const QString &asset, quint16 port, QWidget *parent = nullptr)
        : QWidget(parent)
    {
        setWindowTitle(QString::fromUtf8("2.5D 怀念影像"));
        setStyleSheet("background:#000;");
        setAttribute(Qt::WA_OpaquePaintEvent);

        m_gl = new DibrWidget(this);
        QString err;
        if (!m_gl->loadAsset(asset, &err))
            qWarning() << "加载资产失败:" << err;

        m_hud = new QLabel(this);
        m_hud->setStyleSheet(
            "color:#dfe6ee; background:rgba(0,0,0,120);"
            "font:13px 'DejaVu Sans Mono'; padding:6px 10px; border-radius:6px;");
        m_hud->move(16, 16);
        m_hud->raise();

        connect(m_gl, &DibrWidget::statsUpdated, this,
                [this](float, float ex, float ey) {
                    m_eyeX = ex; m_eyeY = ey; updateHud();
                });

        m_timer = new QTimer(this);
        m_timer->setInterval(16);
        connect(m_timer, &QTimer::timeout, this, [this] {
            if (m_tracking && !m_link->active()) {
                m_tracking = false;
                updateHud();
            }
            if (m_auto) {
                const float t = m_clock.elapsed() / 1000.0f;
                m_gl->setEye(0.72f * std::sin(t * 0.85f),
                             0.28f * std::sin(t * 0.52f));
            }
            // 眨眼：随机间隔 2.5~6 s，单次 160 ms（闭 64ms / 停 32ms / 睁 64ms）
            if (m_gl->hasBlink()) {
                const float t = m_clock.elapsed() / 1000.0f;
                if (!m_blinking && t >= m_nextBlink) {
                    m_blinking = true;
                    m_blinkStart = t;
                    m_nextBlink = t + 2.5f + 3.5f * m_rnd();
                }
                if (m_blinking) {
                    const float dt = t - m_blinkStart;
                    const float D = 0.16f;
                    if (dt >= D) {
                        m_blinking = false;
                        m_gl->setBlink(0.0f);
                    } else {
                        const float p = dt / D;
                        float v;
                        if (p < 0.4f) v = p / 0.4f;
                        else if (p < 0.6f) v = 1.0f;
                        else v = 1.0f - (p - 0.6f) / 0.4f;
                        m_gl->setBlink(v);
                    }
                }
            }

            // 无条件重绘：呼吸/微动是逐帧动画，不能只依赖眼动数据到达才重绘。
            // 否则追踪只有 12fps 时，整个画面也跟着掉到 12fps。
            m_gl->update();
        });
        m_timer->start();
        m_clock.start();

        // 眼位追踪（UDP）。收不到数据时自动退回鼠标/自动摆动。
        m_link = new EyeLink(this);
        QString lerr;
        if (!m_link->start(port, &lerr))
            qWarning() << "眼位 UDP 绑定失败:" << lerr << "端口" << port;
        connect(m_link, &EyeLink::eye, this, [this](float ex, float ey) {
            m_tracking = true;
            m_auto = false;
            m_eyeX = ex;
            m_eyeY = ey;
            m_gl->setEye(ex, ey);
            updateHud();
        });
        connect(m_link, &EyeLink::lost, this, [this] {
            if (m_tracking) { m_tracking = false; updateHud(); }
        });

        updateHud();
        m_gl->setFocus();
    }

    DibrWidget *gl() { return m_gl; }
    /// 快照模式必须关掉自动摆动，否则定时器会把 --eye 覆盖掉
    void setAuto(bool a) { m_auto = a; }
    void toggleFullscreen()
    {
        if (isFullScreen()) showNormal();
        else showFullScreen();
    }

protected:
    void resizeEvent(QResizeEvent *) override
    {
        m_gl->setGeometry(rect());
        m_hud->raise();
    }

    void mouseMoveEvent(QMouseEvent *e) override
    {
        if (m_tracking) return;      // 追踪中，鼠标不抢占
        m_auto = false;
        const float nx = (e->pos().x() / float(width())) * 2.0f - 1.0f;
        const float ny = (e->pos().y() / float(height())) * 2.0f - 1.0f;
        m_gl->setEye(nx * 0.5f, ny * 0.35f);
        updateHud();
    }

    void keyPressEvent(QKeyEvent *e) override
    {
        switch (e->key()) {
        case Qt::Key_Space: m_auto = !m_auto; break;
        case Qt::Key_H: m_showHud = !m_showHud; m_hud->setVisible(m_showHud); break;
        case Qt::Key_F: toggleFullscreen(); break;
        case Qt::Key_S: m_gl->setStyle((m_gl->style() + 1) % 3); break;
        case Qt::Key_Plus:
        case Qt::Key_Equal: m_gl->setGain(m_gl->gain() + 5.0f); break;
        case Qt::Key_Minus: m_gl->setGain(m_gl->gain() - 5.0f); break;
        case Qt::Key_BracketRight: m_gl->setFocusDepth(m_gl->focus() * 1.1f); break;
        case Qt::Key_BracketLeft: m_gl->setFocusDepth(m_gl->focus() / 1.1f); break;
        case Qt::Key_Q:
        case Qt::Key_Escape: close(); return;
        default: QWidget::keyPressEvent(e); return;
        }
        updateHud();
    }

private:
    void updateHud()
    {
        m_hud->setText(QString::fromUtf8(
            "2.5D 怀念影像  |  %1 fps   风格 %8\n"
            "视差 %2   焦点 %3 m   分割 %4 m\n"
            "眼点 (%5, %6)   %7\n"
            "鼠标=视点  空格=自动  S=风格  H=HUD  F=全屏  +/-=视差  [/]=焦点  Q=退出")
            .arg(double(m_gl->fps()), 0, 'f', 1)
            .arg(double(m_gl->gain()), 0, 'f', 0)
            .arg(double(m_gl->focus()), 0, 'f', 2)
            .arg(double(m_gl->split()), 0, 'f', 2)
            .arg(double(m_eyeX), 0, 'f', 2)
            .arg(double(m_eyeY), 0, 'f', 2)
            .arg(m_tracking ? QString::fromUtf8("眼动追踪")
                            : (m_auto ? QString::fromUtf8("自动摆动")
                                      : QString::fromUtf8("鼠标")))
            .arg(m_gl->style() == 0 ? QString::fromUtf8("清晰")
                                    : (m_gl->style() == 1 ? QString::fromUtf8("回忆")
                                                          : QString::fromUtf8("魂影"))));
        m_hud->adjustSize();
    }

    DibrWidget *m_gl = nullptr;
    QLabel *m_hud = nullptr;
    QTimer *m_timer = nullptr;
    QElapsedTimer m_clock;
    bool m_auto = true;
    bool m_showHud = true;
    float m_eyeX = 0.0f;
    float m_eyeY = 0.0f;
    bool m_blinking = false;
    float m_blinkStart = 0.0f;
    float m_nextBlink = 2.0f;
    float m_rnd() {
        // 轻量线性同余，够用
        m_seed = m_seed * 1103515245u + 12345u;
        return float((m_seed >> 16) & 0x7FFF) / 32767.0f;
    }
    unsigned int m_seed = 20261122u;
    EyeLink *m_link = nullptr;
    bool m_tracking = false;
};

int main(int argc, char **argv)
{
    QApplication app(argc, argv);
    QCommandLineParser p;
    p.addHelpOption();
    QCommandLineOption optAsset({"a", "asset"}, "资产目录", "dir", "../assets/demo");
    QCommandLineOption optSnap({"s", "snapshot"}, "渲染一帧存 PNG 后退出", "png");
    QCommandLineOption optEye({"e", "eye"}, "快照时的视点 x,y", "x,y", "0,0");
    QCommandLineOption optFull({"f", "fullscreen"}, "全屏启动");
    QCommandLineOption optFocus("focus", "覆盖焦点深度(米)", "m");
    QCommandLineOption optGain("gain", "覆盖视差强度", "px");
    QCommandLineOption optPort("port", "眼位 UDP 端口", "p", "9999");
    QCommandLineOption optStyle("look", "风格 0清晰/1回忆/2魂影", "i", "0");
    QCommandLineOption optBlink("blink", "快照时固定眨眼量 0..1", "v");
    p.addOptions({optAsset, optSnap, optEye, optFull, optFocus, optGain,
                  optPort, optStyle, optBlink});
    p.process(app);

    Viewer v(p.value(optAsset), quint16(p.value(optPort).toUShort()));

    if (p.isSet(optSnap)) {
        const QStringList xy = p.value(optEye).split(',');
        const float ex = xy.size() > 0 ? xy[0].toFloat() : 0.0f;
        const float ey = xy.size() > 1 ? xy[1].toFloat() : 0.0f;
        v.setAuto(false);
        v.gl()->setStyle(p.value(optStyle).toInt());
        if (p.isSet(optBlink)) v.gl()->setBlink(p.value(optBlink).toFloat());
        if (p.isSet(optFocus)) v.gl()->setFocusDepth(p.value(optFocus).toFloat());
        if (p.isSet(optGain)) v.gl()->setGain(p.value(optGain).toFloat());
        v.gl()->setEye(ex, ey);
        v.resize(v.gl()->assetWidth() / 2, v.gl()->assetHeight() / 2);
        v.show();
        QTimer::singleShot(900, &app, [&] {
            QImage img = v.gl()->snapshotFrame();
            const bool ok = img.save(p.value(optSnap));
            qInfo("snapshot %s -> %s (%dx%d)", ok ? "OK" : "FAIL",
                  qPrintable(p.value(optSnap)), img.width(), img.height());
            app.quit();
        });
        return app.exec();
    }

    v.gl()->setStyle(p.value(optStyle).toInt());
    v.resize(1280, 800);
    if (p.isSet(optFull)) v.showFullScreen();
    else v.show();
    return app.exec();
}
