/**
 * 2.5D 怀念影像 —— 完整系统界面
 *
 * 三个页面：
 *   首页   资产列表（缩略图卡片）+ 新建采集
 *   采集页 相机预览 + 开始采集（调用 Python 流水线）+ 日志
 *   观看页 DIBR 渲染 + 眼动追踪 + 三档风格
 *
 * 架构取舍：重活（深度/抠图/标定/追踪）都在 Python 侧，Qt 只做编排与渲染。
 * 板端没有 OpenCV 的 C++ 头文件，相机预览由一个轻量 Python 服务出帧。
 */
#include <QApplication>
#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QMainWindow>
#include <QMessageBox>
#include <QProcess>
#include <QPushButton>
#include <QScrollArea>
#include <QStackedWidget>
#include <QSurfaceFormat>
#include <QTextEdit>
#include <QTimer>
#include <QToolButton>
#include <QVBoxLayout>
#include <QDebug>
#include <QElapsedTimer>
#include <QIcon>
#include <QTextCursor>
#include <cmath>

#include "dibrwidget.h"
#include "eyelink.h"

// ---------------------------------------------------------------- 全局

static QString g_root;          // 仓库根目录
static QString g_preview = "/tmp/memorial_preview.jpg";

/// 删除一个资产目录。抽出来是为了让界面按钮和 --autodelete 自测走同一条路径。
static bool removeAssetDir(const QString &dir)
{
    QDir d(dir);
    if (!d.exists()) return true;
    return d.removeRecursively();
}

static const char *kQss = R"(
QWidget { background:#0e1013; color:#e6e2dc;
          font-family:'Noto Sans CJK SC','Sans'; font-size:16px; }
QLabel#title   { font-size:34px; font-weight:600; color:#f0e6d8; }
QLabel#sub     { font-size:15px; color:#8d8880; }
QToolButton#card {
    background:#171a1f; border:1px solid #262a31; border-radius:14px;
    padding:10px; color:#d8d2c8; font-size:15px;
}
QToolButton#card:hover { border-color:#d8b48a; background:#1d2128; }
QToolButton#newcard {
    background:#14171c; border:2px dashed #3a4049; border-radius:14px;
    color:#8d8880; font-size:17px;
}
QToolButton#newcard:hover { border-color:#d8b48a; color:#d8b48a; }
QToolButton#del {
    background:rgba(14,16,19,190); border:1px solid #3a4049; border-radius:15px;
    color:#c9c3ba; font-size:15px;
}
QToolButton#del:hover { background:#5a2d2d; border-color:#c96a6a; color:#ffd9d9; }
QPushButton {
    background:#1d2128; border:1px solid #2c323b; border-radius:10px;
    padding:12px 26px; color:#e6e2dc; font-size:17px;
}
QPushButton:hover { border-color:#d8b48a; }
QPushButton:disabled { color:#5a5f66; border-color:#22262c; }
QPushButton#primary { background:#3a2f24; border-color:#8a6f4e; color:#f0dcc0; }
QPushButton#primary:hover { background:#4a3b2c; border-color:#d8b48a; }
QPushButton#stylebtn { padding:8px 18px; border-radius:18px; font-size:15px; }
QPushButton#stylebtn:checked { background:#3a2f24; border-color:#d8b48a; color:#f0dcc0; }
QTextEdit { background:#0a0c0f; border:1px solid #22262c; border-radius:8px;
            color:#9aa0a6; font-family:monospace; font-size:13px; }
QLineEdit { background:#14171c; border:1px solid #2c323b; border-radius:8px;
            padding:8px 12px; }
QMessageBox { background:#14171c; }
QMessageBox QLabel { color:#e2ddd5; font-size:16px; }
QScrollArea { border:none; }
QScrollBar:vertical { background:#0e1013; width:10px; }
QScrollBar::handle:vertical { background:#2c323b; border-radius:5px; }
)";

// ---------------------------------------------------------------- 首页

class HomePage : public QWidget
{
    Q_OBJECT
public:
    explicit HomePage(QWidget *parent = nullptr) : QWidget(parent)
    {
        auto *lay = new QVBoxLayout(this);
        lay->setContentsMargins(48, 36, 48, 36);
        lay->setSpacing(22);

        auto *t = new QLabel(QString::fromUtf8("怀念影像"), this);
        t->setObjectName("title");
        lay->addWidget(t);
        m_sub = new QLabel(this);
        m_sub->setObjectName("sub");
        lay->addWidget(m_sub);

        m_scroll = new QScrollArea(this);
        m_scroll->setWidgetResizable(true);
        auto *inner = new QWidget;
        m_grid = new QGridLayout(inner);
        m_grid->setSpacing(20);
        m_grid->setContentsMargins(0, 8, 0, 8);
        m_grid->setAlignment(Qt::AlignLeft | Qt::AlignTop);
        m_scroll->setWidget(inner);
        lay->addWidget(m_scroll, 1);
    }

    void refresh()
    {
        // 清空
        while (QLayoutItem *it = m_grid->takeAt(0)) {
            if (it->widget()) it->widget()->deleteLater();
            delete it;
        }
        m_assets.clear();

        QDir base(g_root + "/assets");
        const auto dirs = base.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name);
        int col = 0, row = 0;
        for (const QFileInfo &fi : dirs) {
            const QString meta = fi.absoluteFilePath() + "/meta.json";
            if (!QFileInfo::exists(meta) || !QFileInfo::exists(fi.absoluteFilePath() + "/rgb.png"))
                continue;
            m_grid->addWidget(makeCell(fi.fileName(), fi.absoluteFilePath()), row, col);
            if (++col >= 3) { col = 0; ++row; }
        }
        m_grid->addWidget(makeNewCard(), row, col);

        m_sub->setText(QString::fromUtf8("共 %1 个影像 · 站到镜头前，影像会回望你")
                       .arg(m_assets.size()));
    }

signals:
    void openAsset(const QString &dir, const QString &name);
    void newCapture();

private:
    QWidget *makeCell(const QString &name, const QString &dir)
    {
        auto *cell = new QWidget;
        cell->setFixedSize(344, 240);

        auto *b = new QToolButton(cell);
        b->setObjectName("card");
        b->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
        b->setIconSize(QSize(320, 180));
        b->setGeometry(0, 0, 344, 240);
        b->setText(name);
        QPixmap pm(dir + "/rgb.png");
        if (!pm.isNull())
            b->setIcon(QIcon(pm.scaled(320, 180, Qt::KeepAspectRatioByExpanding,
                                       Qt::SmoothTransformation)));
        m_assets << dir;
        connect(b, &QToolButton::clicked, this, [this, dir, name] {
            emit openAsset(dir, name);
        });

        // 右上角删除键：先确认，再把整个目录删掉（不可恢复）
        auto *del = new QToolButton(cell);
        del->setObjectName("del");
        del->setText(QString::fromUtf8("✕"));
        del->setFixedSize(30, 30);
        del->setGeometry(344 - 37, 7, 30, 30);
        del->setToolTip(QString::fromUtf8("删除这个影像"));
        del->setProperty("assetName", name);
        del->raise();
        connect(del, &QToolButton::clicked, this, [this, dir, name] {
            QMessageBox box(QMessageBox::Warning, QString::fromUtf8("删除影像"),
                            QString::fromUtf8("确定要删除「%1」吗？\n\n").arg(name)
                                + QString::fromUtf8("图像、深度、标定数据会一起删除，")
                                + QString::fromUtf8("无法恢复，需要重新采集。"),
                            QMessageBox::NoButton, this);
            QPushButton *yes = box.addButton(QString::fromUtf8("删除"),
                                             QMessageBox::DestructiveRole);
            QPushButton *cancel = box.addButton(QString::fromUtf8("取消"),
                                                QMessageBox::RejectRole);
            box.setDefaultButton(cancel);      // 手滑回车不会删掉东西
            box.exec();
            if (box.clickedButton() != yes) return;
            if (!removeAssetDir(dir)) {
                QMessageBox::warning(this, QString::fromUtf8("删除失败"),
                                     QString::fromUtf8("无法删除\n%1\n请检查目录权限。").arg(dir));
                return;
            }
            qInfo("asset deleted: %s", qPrintable(dir));
            refresh();
        });
        return cell;
    }

    QToolButton *makeNewCard()
    {
        auto *b = new QToolButton;
        b->setObjectName("newcard");
        b->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
        b->setIconSize(QSize(320, 180));
        b->setFixedSize(344, 240);
        b->setText(QString::fromUtf8("\n＋  采集新影像\n"));
        connect(b, &QToolButton::clicked, this, &HomePage::newCapture);
        return b;
    }

    QGridLayout *m_grid = nullptr;
    QScrollArea *m_scroll = nullptr;
    QLabel *m_sub = nullptr;
    QStringList m_assets;
};

// ---------------------------------------------------------------- 采集页

class CapturePage : public QWidget
{
    Q_OBJECT
public:
    explicit CapturePage(QWidget *parent = nullptr) : QWidget(parent)
    {
        auto *lay = new QVBoxLayout(this);
        lay->setContentsMargins(40, 30, 40, 30);
        lay->setSpacing(18);

        auto *head = new QHBoxLayout;
        auto *t = new QLabel(QString::fromUtf8("采集新影像"), this);
        t->setObjectName("title");
        head->addWidget(t);
        head->addStretch();
        m_back = new QPushButton(QString::fromUtf8("返回"), this);
        head->addWidget(m_back);
        lay->addLayout(head);

        auto *body = new QHBoxLayout;
        body->setSpacing(26);

        // 左：预览
        auto *left = new QVBoxLayout;
        m_prev = new QLabel(QString::fromUtf8("相机启动中…"), this);
        m_prev->setFixedSize(640, 360);
        m_prev->setAlignment(Qt::AlignCenter);
        m_prev->setStyleSheet("background:#05070a; border:1px solid #22262c; border-radius:12px;");
        left->addWidget(m_prev);
        m_hint = new QLabel(this);
        m_hint->setObjectName("sub");
        m_hint->setWordWrap(true);
        left->addWidget(m_hint);
        left->addStretch();
        body->addLayout(left);

        // 右：操作 + 日志
        auto *right = new QVBoxLayout;
        right->setSpacing(12);
        auto *nameLbl = new QLabel(QString::fromUtf8("影像名称"), this);
        right->addWidget(nameLbl);
        m_name = new QLineEdit(QString("portrait_%1")
                               .arg(QDateTime::currentDateTime().toString("MMdd_HHmm")), this);
        right->addWidget(m_name);

        m_go = new QPushButton(QString::fromUtf8("开始采集（约 15 秒）"), this);
        m_go->setObjectName("primary");
        m_go->setMinimumHeight(58);
        right->addWidget(m_go);

        m_log = new QTextEdit(this);
        m_log->setReadOnly(true);
        right->addWidget(m_log, 1);
        body->addLayout(right, 1);

        lay->addLayout(body, 1);

        m_tick = new QTimer(this);
        m_tick->setInterval(90);
        connect(m_tick, &QTimer::timeout, this, &CapturePage::onTick);
        connect(m_back, &QPushButton::clicked, this, [this] { stopAll(); emit cancelled(); });
        connect(m_go, &QPushButton::clicked, this, &CapturePage::startCapture);

        m_preview = new QProcess(this);
        m_preview->setProcessChannelMode(QProcess::MergedChannels);
        m_capture = new QProcess(this);
        // RKNN 运行时每帧都往 stderr 刷 I/W 日志，合并通道会把真正的结果埋掉。
        // 所以只把 stdout 接进日志，stderr 留着失败时取尾巴（traceback）。
        m_capture->setProcessChannelMode(QProcess::SeparateChannels);
        connect(m_capture, &QProcess::readyRead, this, [this] {
            const QString s = QString::fromUtf8(m_capture->readAllStandardOutput());
            if (s.isEmpty()) return;
            m_log->moveCursor(QTextCursor::End);
            m_log->insertPlainText(s);
            m_log->moveCursor(QTextCursor::End);
        });
        connect(m_capture, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
                this, &CapturePage::onCaptureDone);
    }

    ~CapturePage() override { stopAll(); }

    void enter()
    {
        m_log->clear();
        m_log->append(QString::fromUtf8("准备就绪。请站到镜头前约 1.2~1.5 米，正对镜头。"));
        m_hint->setText(QString::fromUtf8(
            "提示：站到画面中央，面向光源，不要背光。采集会自动标定尺度并生成完整资产。"));
        m_go->setEnabled(true);
        startPreview();
    }

    /// kiosk / 自动化测试：指定名字后直接开始采集（--auto-start 用）
    void autoStart(const QString &name)
    {
        m_name->setText(name);
        QTimer::singleShot(600, this, &CapturePage::startCapture);
    }

signals:
    void finished(const QString &name);
    void cancelled();

private:
    void onTick()
    {
        QFileInfo fi(g_preview);
        if (fi.exists() && fi.lastModified() != m_stamp) {
            m_stamp = fi.lastModified();
            QPixmap pm;
            if (pm.load(g_preview)) {
                m_previewGotFrame = true;
                m_prev->setPixmap(pm.scaled(m_prev->size(), Qt::KeepAspectRatio,
                                            Qt::SmoothTransformation));
                return;
            }
        }
        // 相机被别的进程占着（最常见：眼动追踪器、或上一次的 preview_server 没退）
        // 时，preview_server 会直接退出，这里以前只有一块黑框，用户不知道发生了什么。
        if (m_previewGotFrame || m_previewWarned) return;
        const bool died = m_preview->state() == QProcess::NotRunning;
        if (!died && m_previewWait.elapsed() < 2500) return;
        m_previewWarned = true;
        m_prev->setText(QString::fromUtf8(
            "相机不可用\n可能被眼动追踪器或其它程序占用\n\n"
            "先关掉追踪（或运行 pkill -f eye_tracker），\n"
            "再点「返回」重新进入本页"));
        m_prev->setAlignment(Qt::AlignCenter);
        m_log->append(QString::fromUtf8(
            "\n✗ 预览没有画面：相机打不开（通常是被眼动追踪器占用）。"));
        m_log->append(QString::fromUtf8(
            "  查看占用：pgrep -af 'eye_tracker|preview_server'"));
    }

    void startPreview()
    {
        if (m_preview->state() != QProcess::NotRunning) return;
        m_preview->start("python3", {g_root + "/tools/preview_server.py", "--out", g_preview});
        m_tick->start();
        m_previewGotFrame = false;
        m_previewWarned = false;
        m_previewWait.start();
    }

    void stopPreview()
    {
        m_tick->stop();
        if (m_preview->state() != QProcess::NotRunning) {
            m_preview->terminate();
            if (!m_preview->waitForFinished(1500)) m_preview->kill();
        }
        m_prev->setText(QString::fromUtf8("相机已释放"));
        m_prev->setPixmap(QPixmap());
    }

    void stopAll() { stopPreview(); }

    void startCapture()
    {
        const QString name = m_name->text().trimmed();
        if (name.isEmpty()) {
            m_log->append(QString::fromUtf8("✗ 先给这个影像起个名字"));
            return;
        }
        // 采集失败时会把这次新建的目录整个删掉（见 abortCapture）。若允许覆盖
        // 同名资产，一次失败的采集就会顺手删掉用户原来的影像，所以先挡住。
        if (QFileInfo::exists(g_root + "/assets/" + name)) {
            m_log->append(QString::fromUtf8("✗ 资产「%1」已存在，换个名字再采集").arg(name));
            return;
        }
        stopPreview();                      // 采集脚本要独占相机
        m_go->setEnabled(false);
        m_log->clear();
        m_log->append(QString::fromUtf8(">>> 采集与处理中，请保持站姿…\n"));
        m_pendingName = name;
        m_stage = 1;

        QStringList args{g_root + "/tools/mono_calib.py",
                         "--frames", "3", "--write", "--save-asset", name};
        m_capture->start("python3", args);
    }

    /// 采集判定失败：丢弃本次写出的半个资产，把相机交还预览让用户重试。
    /// 目录是本次新建的（startCapture 已挡掉同名），删掉不会误伤旧资产。
    void abortCapture(const QString &why)
    {
        const QString dir = g_root + "/assets/" + m_pendingName;
        if (QFileInfo::exists(dir))
            QDir(dir).removeRecursively();

        m_log->append(QString::fromUtf8("\n✗ %1").arg(why));
        // stderr 只取最后几行：前面是 RKNN 加载噪音，有用的是 traceback 尾巴
        const QStringList err = QString::fromUtf8(m_capture->readAllStandardError())
                                    .split('\n', Qt::SkipEmptyParts);
        if (!err.isEmpty())
            m_log->append(QString::fromUtf8("  %1").arg(err.mid(qMax(0, err.size() - 6)).join('\n')));
        m_log->append(QString::fromUtf8("  本次采集已丢弃，请重新采集。"));

        m_hint->setText(QString::fromUtf8("✗ %1 —— 请重新采集").arg(why));
        m_stage = 0;
        m_go->setEnabled(true);
        startPreview();
    }

    void onCaptureDone(int code, QProcess::ExitStatus)
    {
        // 两段式：1 = 采集+标定，2 = 后处理（眨眼/头部遮罩）
        if (m_stage == 1) {
            if (code != 0) {
                abortCapture(code == 3
                    ? QString::fromUtf8("受采者不在 1.2~1.5 m（没检测到人，或瞳距 < 45 px）")
                    : (code == 4
                       ? QString::fromUtf8("标定质量不足（内点或近距样本不够），未生成资产")
                       : QString::fromUtf8("采集失败（退出码 %1）").arg(code)));
                return;
            }
            m_log->append(QString::fromUtf8("\n>>> 生成眨眼与头部遮罩…\n"));
            m_stage = 2;
            m_capture->start("python3", {g_root + "/tools/finalize_asset.py",
                                         g_root + "/assets/" + m_pendingName});
            return;
        }
        if (m_stage == 2) {
            if (code != 0) {
                // 3 = 瞳距守卫（离得太远/不在画面中央），2 = 没检测到双眼。
                // 以前这里只写"后处理有告警，资产仍可渲染"，于是废资产照样进首页、
                // 还直接跳进观看页——守卫等于白加。
                abortCapture(code == 3
                    ? QString::fromUtf8("受采者离得太远或不在画面中央（瞳距 < 45 px），"
                                        "眨眼与头部遮罩未生成")
                    : (code == 2
                       ? QString::fromUtf8("没检测到双眼，眨眼与头部遮罩未生成")
                       : QString::fromUtf8("后处理失败（退出码 %1）").arg(code)));
                return;
            }
            m_log->append(QString::fromUtf8("\n✓ 完成：rgb/depth/matte/bg/blink/head/meta 已齐"));
            m_stage = 0;
            emit finished(m_pendingName);
        }
    }

    QLabel *m_prev = nullptr;
    QLabel *m_hint = nullptr;
    QPushButton *m_back = nullptr;
    QPushButton *m_go = nullptr;
    QLineEdit *m_name = nullptr;
    QTextEdit *m_log = nullptr;
    QTimer *m_tick = nullptr;
    QProcess *m_preview = nullptr;
    QElapsedTimer m_previewWait;
    bool m_previewGotFrame = false;
    bool m_previewWarned = false;
    QProcess *m_capture = nullptr;
    QDateTime m_stamp;
    QString m_pendingName;
    int m_stage = 0;
};

// ---------------------------------------------------------------- 观看页

class StagePage : public QWidget
{
    Q_OBJECT
public:
    explicit StagePage(QWidget *parent = nullptr) : QWidget(parent)
    {
        m_gl = new DibrWidget(this);

        m_top = new QWidget(this);
        m_top->setStyleSheet("background:rgba(8,10,13,150); border-radius:14px;");
        auto *tl = new QHBoxLayout(m_top);
        tl->setContentsMargins(14, 8, 14, 8);
        m_back = new QPushButton(QString::fromUtf8("← 返回"), m_top);
        m_back->setFlat(true);
        m_title = new QLabel(m_top);
        m_title->setStyleSheet("font-size:18px; color:#f0e6d8; background:transparent;");
        m_stat = new QLabel(m_top);
        m_stat->setStyleSheet("font-size:14px; color:#8d8880; background:transparent;");
        tl->addWidget(m_back);
        tl->addSpacing(12);
        tl->addWidget(m_title);
        tl->addStretch();
        tl->addWidget(m_stat);

        m_bottom = new QWidget(this);
        m_bottom->setStyleSheet("background:rgba(8,10,13,150); border-radius:14px;");
        auto *bl = new QHBoxLayout(m_bottom);
        bl->setContentsMargins(16, 8, 16, 8);
        const char *names[3] = {"清晰", "回忆", "魂影"};
        for (int i = 0; i < 3; ++i) {
            auto *b = new QPushButton(QString::fromUtf8(names[i]), m_bottom);
            b->setObjectName("stylebtn");
            b->setCheckable(true);
            b->setChecked(i == 1);
            m_styleBtns[i] = b;
            connect(b, &QPushButton::clicked, this, [this, i] {
                for (int k = 0; k < 3; ++k) m_styleBtns[k]->setChecked(k == i);
                m_gl->setStyle(i);
            });
            bl->addWidget(b);
        }
        bl->addStretch();
        m_trackBtn = new QPushButton(QString::fromUtf8("开启眼动追踪"), m_bottom);
        m_trackBtn->setCheckable(true);
        connect(m_trackBtn, &QPushButton::toggled, this, &StagePage::toggleTracking);
        bl->addWidget(m_trackBtn);

        connect(m_back, &QPushButton::clicked, this, [this] {
            toggleTracking(false);
            m_trackBtn->setChecked(false);
            emit back();
        });

        m_link = new EyeLink(this);
        m_link->start(9999);
        connect(m_link, &EyeLink::eye, this, [this](float ex, float ey) {
            m_tracking = true;
            m_gl->setEye(ex, ey);
            updateStat();
        });
        connect(m_link, &EyeLink::lost, this, [this] { m_tracking = false; updateStat(); });

        m_timer = new QTimer(this);
        m_timer->setInterval(16);
        connect(m_timer, &QTimer::timeout, this, [this] { tick(); });
        m_timer->start();
        m_clock.start();

        m_trackerProc = new QProcess(this);
        m_trackerProc->setProcessChannelMode(QProcess::MergedChannels);
    }

    ~StagePage() override { toggleTracking(false); }

    void open(const QString &dir, const QString &name)
    {
        QString err;
        if (!m_gl->loadAsset(dir, &err))
            qWarning() << "资产加载失败:" << err;
        m_gl->setStyle(1);
        for (int i = 0; i < 3; ++i) m_styleBtns[i]->setChecked(i == 1);
        m_title->setText(name);
        m_assetDir = dir;
        m_tracking = false;
        m_gaze = 0.0f;
        m_auto = true;
        updateStat();
    }

    /// 进采集页前必须停掉追踪：眼动追踪器独占相机，不停的话采集页预览打不开
    /// （实测踩过：观看页开着追踪 → 回首页 → 点采集 → 预览一片黑）
    void stopTracking()
    {
        if (m_trackBtn && m_trackBtn->isChecked()) m_trackBtn->setChecked(false);
        toggleTracking(false);
    }

signals:
    void back();

protected:
    void resizeEvent(QResizeEvent *) override
    {
        m_gl->setGeometry(rect());
        m_top->setGeometry(24, 20, width() - 48, 54);
        m_bottom->setGeometry(24, height() - 74, width() - 48, 54);
        m_top->raise();
        m_bottom->raise();
    }

private:
    void tick()
    {
        // 无追踪时缓慢自动摆动，便于演示
        if (!m_tracking && m_auto) {
            const float t = m_clock.elapsed() / 1000.0f;
            m_gl->setEye(0.55f * std::sin(t * 0.7f), 0.20f * std::sin(t * 0.45f));
        }
        // 回望强度
        const float target = m_tracking ? 1.0f : 0.0f;
        m_gaze += (target - m_gaze) * 0.06f;
        m_gl->setGaze(m_gaze);
        // 眨眼
        const float t = m_clock.elapsed() / 1000.0f;
        if (!m_blinking && t >= m_nextBlink) {
            m_blinking = true; m_blinkStart = t;
            m_seed = m_seed * 1103515245u + 12345u;
            m_nextBlink = t + 2.5f + 3.5f * float((m_seed >> 16) & 0x7FFF) / 32767.0f;
        }
        if (m_blinking) {
            const float dt = t - m_blinkStart;
            if (dt >= 0.16f) { m_blinking = false; m_gl->setBlink(0.0f); }
            else {
                const float p = dt / 0.16f;
                float v = (p < 0.4f) ? p / 0.4f : (p < 0.6f ? 1.0f : 1.0f - (p - 0.6f) / 0.4f);
                m_gl->setBlink(v);
            }
        }
        m_gl->update();
    }

    void updateStat()
    {
        m_stat->setText(m_tracking ? QString::fromUtf8("● 眼动追踪中")
                                   : QString::fromUtf8("○ 未追踪 · 自动演示"));
        m_stat->setStyleSheet(m_tracking
            ? "font-size:14px; color:#8fce8f; background:transparent;"
            : "font-size:14px; color:#8d8880; background:transparent;");
    }

    void toggleTracking(bool on)
    {
        if (on) {
            m_auto = false;
            if (m_trackerProc->state() == QProcess::NotRunning) {
                m_trackerProc->start("python3", {g_root + "/track/eye_tracker.py"});
            }
        } else {
            if (m_trackerProc->state() != QProcess::NotRunning) {
                m_trackerProc->terminate();
                if (!m_trackerProc->waitForFinished(1500)) m_trackerProc->kill();
            }
            m_tracking = false;
            m_auto = true;
            updateStat();
        }
    }

    DibrWidget *m_gl = nullptr;
    QWidget *m_top = nullptr, *m_bottom = nullptr;
    QPushButton *m_back = nullptr, *m_trackBtn = nullptr;
    QPushButton *m_styleBtns[3] = {nullptr, nullptr, nullptr};
    QLabel *m_title = nullptr, *m_stat = nullptr;
    EyeLink *m_link = nullptr;
    QProcess *m_trackerProc = nullptr;
    QTimer *m_timer = nullptr;
    QElapsedTimer m_clock;
    QString m_assetDir;
    bool m_tracking = false, m_auto = true;
    float m_gaze = 0.0f;
    bool m_blinking = false;
    float m_blinkStart = 0.0f, m_nextBlink = 2.0f;
    unsigned int m_seed = 20261123u;
};

// ---------------------------------------------------------------- 主窗口

class Studio : public QMainWindow
{
    Q_OBJECT
public:
    Studio()
    {
        setWindowTitle(QString::fromUtf8("2.5D 怀念影像"));
        m_stack = new QStackedWidget(this);
        setCentralWidget(m_stack);

        m_home = new HomePage;
        m_cap = new CapturePage;
        m_stage = new StagePage;
        m_stack->addWidget(m_home);
        m_stack->addWidget(m_cap);
        m_stack->addWidget(m_stage);

        connect(m_home, &HomePage::openAsset, this, [this](const QString &d, const QString &n) {
            m_stage->open(d, n);
            m_stack->setCurrentWidget(m_stage);
        });
        connect(m_home, &HomePage::newCapture, this, [this] {
            m_stage->stopTracking();        // 相机要让出来，否则预览打不开
            m_cap->enter();
            m_stack->setCurrentWidget(m_cap);
        });
        connect(m_cap, &CapturePage::cancelled, this, [this] {
            m_home->refresh();
            m_stack->setCurrentWidget(m_home);
        });
        connect(m_cap, &CapturePage::finished, this, [this](const QString &n) {
            m_home->refresh();
            m_stage->open(g_root + "/assets/" + n, n);
            m_stack->setCurrentWidget(m_stage);
        });
        connect(m_stage, &StagePage::back, this, [this] {
            m_home->refresh();
            m_stack->setCurrentWidget(m_home);
        });

        m_home->refresh();
    }

    /// 直接进入某个页面（kiosk 启动 / 自动化测试用）
    void goHome() { m_home->refresh(); m_stack->setCurrentWidget(m_home); }
    void goCapture() { m_stage->stopTracking(); m_cap->enter(); m_stack->setCurrentWidget(m_cap); }
    /// kiosk / 自动化测试：进采集页并立刻开始（--capture --auto-start <名字>）
    void autoStartCapture(const QString &name)
    {
        m_cap->enter();
        m_stack->setCurrentWidget(m_cap);
        m_cap->autoStart(name);
    }
    void goStage(const QString &dir, const QString &name)
    {
        m_stage->open(dir, name);
        m_stack->setCurrentWidget(m_stage);
    }

private:
    QStackedWidget *m_stack = nullptr;
    HomePage *m_home = nullptr;
    CapturePage *m_cap = nullptr;
    StagePage *m_stage = nullptr;
};

int main(int argc, char **argv)
{
    // 垂直同步：不锁的话画面在动的时候会出现水平撕裂（实测反馈）
    QSurfaceFormat fmt;
    fmt.setSwapInterval(1);
    QSurfaceFormat::setDefaultFormat(fmt);
    QApplication app(argc, argv);
    QCoreApplication::setApplicationName("memorial-studio");

    g_root = QDir(QCoreApplication::applicationDirPath() + "/../..").canonicalPath();

    app.setStyleSheet(kQss);
    Studio w;
    w.resize(1280, 720);
    // 默认最大化：板子屏幕是 1280x720，还带顶栏和 dock，窗口化只剩 586 px 高，
    // 16:9 的资产只能按高度铺，左右就空出黑边（用户反馈"影像变窄了"）。
    bool fullscreen = false;
    for (int i = 1; i < argc; ++i)
        if (QString(argv[i]) == "--fullscreen") fullscreen = true;
    if (fullscreen) w.showFullScreen();
    else w.showMaximized();

    // --asset <名称|路径>  直接进入观看页；--capture 直接进入采集页
    const QStringList args = app.arguments();
    const int ai = args.indexOf("--asset");
    if (ai >= 0 && ai + 1 < args.size()) {
        QString a = args[ai + 1];
        QString dir = a.startsWith("/") ? a : (g_root + "/assets/" + a);
        w.goStage(dir, QFileInfo(dir).fileName());
    } else if (args.contains("--capture")) {
        // --capture --auto-start <名字>：kiosk / 自动化测试用，进页后自动开拍
        const int ni = args.indexOf("--auto-start");
        if (ni >= 0 && ni + 1 < args.size()) w.autoStartCapture(args[ni + 1]);
        else w.goCapture();
    }

    // --autodelete <名称> [--autodelete-delay ms]：点一次删除键并自动确认，
    // 结束后按目录是否真的消失给出退出码（0 成功 / 5 失败），用于自动化验证。
    const int di = args.indexOf("--autodelete");
    if (di >= 0 && di + 1 < args.size()) {
        const QString name = args[di + 1];
        int delay = 300;
        const int dli = args.indexOf("--autodelete-delay");
        if (dli >= 0 && dli + 1 < args.size()) delay = args[dli + 1].toInt();
        if (delay <= 0) delay = 300;

        QTimer::singleShot(800, &w, [&w, name, delay] {
            QToolButton *target = nullptr;
            for (QToolButton *b : w.findChildren<QToolButton *>("del"))
                if (b->property("assetName").toString() == name) target = b;
            if (!target) {
                qWarning("AUTODELETE %s : FAIL(no delete button)", qPrintable(name));
                QCoreApplication::exit(6);
                return;
            }
            // 模态框弹出后会阻塞在嵌套事件循环里，所以先用定时器挂好"自动点 Yes"
            QElapsedTimer t0;
            t0.start();
            auto *ans = new QTimer(qApp);
            ans->setInterval(50);
            QObject::connect(ans, &QTimer::timeout, [ans, delay, t0] {
                auto *mb = qobject_cast<QMessageBox *>(QApplication::activeModalWidget());
                if (!mb) return;
                if (t0.elapsed() < delay) return;
                fprintf(stderr, "[autodelete] modal found, clicking Yes\n");
                fflush(stderr);
                ans->stop();
                QAbstractButton *yes = nullptr;
                for (auto *b : mb->buttons())
                    if (mb->buttonRole(b) == QMessageBox::DestructiveRole) yes = b;
                if (yes) yes->click();
                else mb->done(QMessageBox::Yes);
            });
            ans->start();
            fprintf(stderr, "[autodelete] clicking delete button for %s\n", qPrintable(name));
            fflush(stderr);
            target->click();
            ans->stop();
            fprintf(stderr, "[autodelete] dialog closed, dir exists=%d\n",
                    int(QDir(g_root + "/assets/" + name).exists()));
            fflush(stderr);
            const bool gone = !QDir(g_root + "/assets/" + name).exists();
            qInfo("AUTODELETE %s : %s", qPrintable(name),
                  gone ? "OK(dir removed)" : "FAIL(dir still exists)");
            QCoreApplication::exit(gone ? 0 : 5);
        });
    }

    // --autoswitch a,b：自动走一遍"看 a → 返回 → 看 b"，用于复现切换素材
    const int si = args.indexOf("--autoswitch");
    if (si >= 0 && si + 1 < args.size()) {
        const QStringList two = args[si + 1].split(',');
        if (two.size() >= 2) {
            QTimer::singleShot(1500, &w, [&w, two] {
                w.goStage(g_root + "/assets/" + two[0], two[0]);
            });
            QTimer::singleShot(6000, &w, [&w] { w.goHome(); });
            QTimer::singleShot(8000, &w, [&w, two] {
                w.goStage(g_root + "/assets/" + two[1], two[1]);
            });
        }
    }
    return app.exec();
}

#include "studio.moc"
