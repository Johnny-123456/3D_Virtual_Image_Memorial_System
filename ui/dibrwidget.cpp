#include "dibrwidget.h"

#include <QDebug>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

#include <cmath>
#include <vector>

// 掩码内的盒式模糊（积分图实现，O(1)/像素），定义见文件后半
static void maskedBlur(std::vector<float> &d, int w, int h,
                       const std::vector<uchar> &mask, int r);

#ifndef GL_R32F
#define GL_R32F 0x822E
#endif
#ifndef GL_RED
#define GL_RED 0x1903
#endif
#ifndef GL_CLAMP_TO_EDGE
#define GL_CLAMP_TO_EDGE 0x812F
#endif

static const char *kVert = R"(
attribute vec2 aPos;
attribute vec2 aUV;

uniform sampler2D uDepthTex;
uniform float uGain, uZFoc, uEyeX, uEyeY, uHalfW, uHalfH, uFitX, uFitY;
uniform float uTime, uMaxShift, uPx;
uniform sampler2D uHeadMask;
uniform float uGaze, uGazeGain;

varying vec2 vUV;

void main() {
    float z = texture2D(uDepthTex, aUV).r;
    z = max(z, 0.15);

    // 视差与 1/z 之差成正比：焦点处为 0，更近的前移、更远的后移
    float p = (1.0 / uZFoc) - (1.0 / z);
    vec2 disp = vec2(uEyeX, uEyeY) * uGain * p;
    disp = clamp(disp, vec2(-uMaxShift), vec2(uMaxShift));

    vec2 pos = aPos + disp;

    // 头部区域的两层微动，都按 head.png 加权：
    //  1) 回望：轻微前倾。幅度必须小——head.png 是边缘柔和衰减的椭圆，位移按它
    //     加权意味着脸内部不同位置走不同距离，幅度一大就是剪切，脸越转越歪。
    //     真正的转头要靠神经重演（LivePortrait），2.5D 网格做不到。
    //  2) 点头：极缓慢的上下浮动，幅度 1.8px、周期约 8.6 s，纯粹为了"别太死板"。
    // 注意：网格坐标是"资产像素"，所以凡是绝对值（px）的动效都要乘 uPx
    // 归一到 1272 px 的参照资产上，否则小图会被放大成夸张的晃动。
    float hm = texture2D(uHeadMask, aUV).r;
    if (uGaze > 0.001) {
        pos += vec2(uEyeX, uEyeY) * uGaze * uGazeGain * uPx * hm;
    }
    pos.y += 1.8 * uPx * sin(uTime * 0.73 + 1.7) * hm;

    // 呼吸：以画面中心为基准的极轻微纵向缩放 + 横向漂移
    pos.y *= 1.0 + 0.0026 * sin(uTime * 1.6);
    pos.x += 0.8 * uPx * sin(uTime * 0.7);

    vUV = aUV;
    gl_Position = vec4((pos.x / uHalfW) * uFitX,
                       (pos.y / uHalfH) * uFitY, 0.0, 1.0);
}
)";

static const char *kFrag = R"(
uniform sampler2D uTex;
uniform sampler2D uTexBlur;
uniform sampler2D uDepthTex;
uniform sampler2D uMatte;
uniform sampler2D uGlow;
uniform sampler2D uTexBlink;
uniform int uHasBlink;
uniform float uBlink;
uniform int uHasMatte;
uniform float uSplit;
uniform float uGhostAlpha;
uniform float uGlowAmt;
uniform float uGrade;
uniform int uPass;
uniform sampler2D uEyeMask;      // 两眼位置的小椭圆，做"眼神微动"
uniform float uHasEye;
uniform vec2 uEyeShift;          // UV 偏移（由 CPU 按像素换算好）

varying vec2 vUV;

// 回忆调色：轻微降饱和 + 偏暖 + 提黑位（褪色胶片感）
vec3 grade(vec3 c) {
    float l = dot(c, vec3(0.299, 0.587, 0.114));
    c = mix(vec3(l), c, 1.0 - 0.18 * uGrade);
    c *= mix(vec3(1.0), vec3(1.07, 1.00, 0.93), uGrade);
    c = mix(c, c * 0.93 + 0.035, uGrade);
    return c;
}

void main() {
    vec2 d = vUV - 0.5;
    float vig = 1.0 - dot(d, d) * (0.45 + 0.25 * uGrade);

    if (uPass == 0) {
        vec3 bg = texture2D(uTexBlur, vUV).rgb;
        bg = grade(bg) * vig * 0.9;
        gl_FragColor = vec4(bg, 1.0);

    } else if (uPass == 1) {
        // 眼神微动：只在两眼的小椭圆内把采样点挪一点点（1~2 px），
        // 看上去是眼球在转动，而不是整张脸在动。
        vec2 uv = vUV;
        if (uHasEye > 0.5) {
            float em = texture2D(uEyeMask, vUV).r;
            uv += uEyeShift * em;
        }
        vec3 c = texture2D(uTex, uv).rgb;
        if (uHasBlink == 1 && uBlink > 0.001) {
            vec3 c2 = texture2D(uTexBlink, uv).rgb;
            c = mix(c, c2, uBlink);
        }
        c = grade(c) * vig;

        float alpha;
        if (uHasMatte == 1) {
            alpha = texture2D(uMatte, vUV).r;
        } else {
            float z = texture2D(uDepthTex, vUV).r;
            float t = clamp((z - uSplit * 0.80) / max(uSplit * 0.40, 1e-3), 0.0, 1.0);
            alpha = 1.0 - t;
        }
        alpha *= uGhostAlpha;
        if (alpha <= 0.002) discard;
        gl_FragColor = vec4(c, alpha);

    } else {
        // pass 2：边缘辉光，叠加混合（glBlendFunc(GL_ONE, GL_ONE)）
        float m = texture2D(uMatte, vUV).r;
        float g = texture2D(uGlow, vUV).r;
        // 只保留明显的外扩带。直接 pow(rim,0.6)*2.2 会把 8-bit 量化的
        // 微小残差放大成与网格对齐的竖条（实测踩过），这里用 smoothstep
        // 做门限：低于 0.05 的一律不发光，背景不会被叠加到过曝。
        float rim = clamp(g - m, 0.0, 1.0);
        // 直接丢弃弱残差：背景处 blur(matte)-matte 只有 1/255 量级的量化残差，
        // 一旦参与叠加就会被网格插值放大成竖条带（实测踩过两次）。
        if (rim < 0.10) discard;
        rim = smoothstep(0.10, 0.45, rim) * uGlowAmt;
        vec3 warm = vec3(1.0, 0.87, 0.70);
        gl_FragColor = vec4(warm * min(rim, 0.55), 1.0);
    }
}
)";

DibrWidget::DibrWidget(QWidget *parent) : QOpenGLWidget(parent)
{
    m_clock.start();
    m_fpsTimer.start();
    setMouseTracking(true);
}

void DibrWidget::setStyle(int idx)
{
    m_style = qBound(0, idx, 2);
    switch (m_style) {
    case 0:  // 清晰
        m_ghostAlpha = 1.00f; m_glowAmt = 0.30f; m_grade = 0.25f; break;
    case 1:  // 回忆
        m_ghostAlpha = 1.00f; m_glowAmt = 0.55f; m_grade = 0.75f; break;
    default: // 魂影
        m_ghostAlpha = 0.78f; m_glowAmt = 0.95f; m_grade = 1.00f; break;
    }
    update();
}

void DibrWidget::setBlink(float v)
{
    m_blink = qBound(0.0f, v, 1.0f);
    update();
}

void DibrWidget::setGaze(float v)
{
    m_gaze = qBound(0.0f, v, 1.0f);
    update();
}

void DibrWidget::setEye(float ex, float ey)
{
    m_eyeX = qBound(-1.0f, ex, 1.0f);
    m_eyeY = qBound(-1.0f, ey, 1.0f);
    update();
}

bool DibrWidget::loadAsset(const QString &dir, QString *err)
{
    auto fail = [&](const QString &m) {
        if (err) *err = m;
        m_ok = false;
        return false;
    };

    QFile mf(dir + "/meta.json");
    if (!mf.open(QIODevice::ReadOnly)) return fail("打不开 meta.json");
    QJsonObject meta = QJsonDocument::fromJson(mf.readAll()).object();
    m_w = meta.value("width").toInt();
    m_h = meta.value("height").toInt();
    m_zFocus = float(meta.value("z_focus").toDouble(5.0));
    m_zSubject = float(meta.value("z_subject").toDouble(m_zFocus));
    if (m_zSubject <= 0.05f) m_zSubject = m_zFocus;
    m_zMin = float(meta.value("z_min").toDouble(0.3));
    m_zMax = float(meta.value("z_max").toDouble(10.0));
    // 分割面优先取资产里的 z_split：只有比它更近的才走全视差
    m_split = qBound(m_zMin,
                     float(meta.value("z_split").toDouble(m_zFocus * 1.6)),
                     m_zMax);
    if (m_w <= 0 || m_h <= 0) return fail("meta.json 宽高无效");

    m_img = QImage(dir + "/rgb.png");
    if (m_img.isNull()) return fail("打不开 rgb.png");
    m_img = m_img.convertToFormat(QImage::Format_RGBA8888);
    // 预生成虚化背景：缩小再放大即可得到柔和的低频版本
    m_blur = m_img.scaled(qMax(8, m_img.width() / 10), qMax(8, m_img.height() / 10),
                          Qt::IgnoreAspectRatio, Qt::SmoothTransformation)
                 .scaled(m_img.width(), m_img.height(),
                         Qt::IgnoreAspectRatio, Qt::SmoothTransformation);

    // 背景板：优先用 tools/make_bg.py 生成的 bg.png（已把主体挖掉并修补）。
    // 没有就退回模糊原图——那样主体移开后会露出自己的影子（鬼影）。
    QImage bgImg(dir + "/bg.png");
    if (!bgImg.isNull()) {
        m_bg = bgImg.convertToFormat(QImage::Format_RGBA8888)
                       .scaled(m_img.width(), m_img.height(),
                               Qt::IgnoreAspectRatio, Qt::SmoothTransformation)
                       .scaled(qMax(8, m_img.width() / 8), qMax(8, m_img.height() / 8),
                               Qt::IgnoreAspectRatio, Qt::SmoothTransformation)
                       .scaled(m_img.width(), m_img.height(),
                               Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
    } else {
        m_bg = m_blur;
    }
    if (m_img.width() != m_w || m_img.height() != m_h) {
        m_w = m_img.width();
        m_h = m_img.height();
    }

    QImage matteImg(dir + "/matte.png");
    if (!matteImg.isNull()) {
        m_matte = matteImg.convertToFormat(QImage::Format_Grayscale8)
                          .scaled(m_w, m_h, Qt::IgnoreAspectRatio,
                                  Qt::SmoothTransformation);
        m_hasMatte = true;
    } else {
        m_hasMatte = false;
    }

    QImage blinkImg(dir + "/blink.png");
    if (!blinkImg.isNull()) {
        m_blinkImg = blinkImg.convertToFormat(QImage::Format_RGBA8888)
                             .scaled(m_w, m_h, Qt::IgnoreAspectRatio,
                                     Qt::SmoothTransformation);
        m_hasBlink = true;
    } else {
        m_hasBlink = false;
    }

    QImage headImg(dir + "/head.png");
    if (!headImg.isNull()) {
        m_headMask = headImg.convertToFormat(QImage::Format_Grayscale8)
                             .scaled(m_w, m_h, Qt::IgnoreAspectRatio,
                                     Qt::SmoothTransformation);
        m_hasHead = true;
    } else {
        m_hasHead = false;
    }

    // 眼睛位置来自 meta.json（采集时 finalize_asset 写入）。有的话建两个小椭圆，
    // 渲染端在椭圆内轻微挪动采样点 = 眼球在转，让画面不那么"死"。
    m_hasEyes = false;
    const QJsonObject eyes = meta.value("eyes").toObject();
    if (eyes.contains("left") && eyes.contains("right")) {
        const QJsonArray l = eyes.value("left").toArray();
        const QJsonArray r = eyes.value("right").toArray();
        if (l.size() == 2 && r.size() == 2) {
            const float lx = float(l[0].toDouble()), ly = float(l[1].toDouble());
            const float rx = float(r[0].toDouble()), ry = float(r[1].toDouble());
            const float iod = float(std::hypot(rx - lx, ry - ly));
            const float ex = qMax(4.0f, iod * 0.26f), ey = qMax(3.0f, iod * 0.17f);
            QImage em(m_w, m_h, QImage::Format_Grayscale8);
            em.fill(0);
            for (int py = 0; py < m_h; ++py) {
                uchar *row = em.scanLine(py);
                for (int px = 0; px < m_w; ++px) {
                    float v = 0.0f;
                    for (const QPointF &c : {QPointF(lx, ly), QPointF(rx, ry)}) {
                        const float dx = (px - c.x()) / ex, dy = (py - c.y()) / ey;
                        const float d2 = dx * dx + dy * dy;
                        if (d2 < 1.0f) v = qMax(v, 1.0f - d2);   // 软椭圆
                    }
                    row[px] = uchar(qBound(0.0f, v, 1.0f) * 255.0f);
                }
            }
            m_eyeMask = em;
            m_hasEyes = true;
        }
    }

    if (m_hasMatte) {
        // 辉光贴图 = matte 的高斯模糊版。必须全分辨率模糊：
        // 之前先缩到 1/12 再放大，blur(matte)-matte 的差值是 8-bit 量化台阶，
        // 经 GL_ONE 叠加后在暗背景上变成竖条带（实测踩坑）。
        // 这里用 3 次全分辨率下采样/上采样的盒式模糊逼近高斯，台阶只有 1px。
        QImage g = m_matte;
        for (int pass = 0; pass < 3; ++pass) {
            int w2 = qMax(8, g.width() / 2), h2 = qMax(8, g.height() / 2);
            g = g.scaled(w2, h2, Qt::IgnoreAspectRatio, Qt::SmoothTransformation)
                 .scaled(m_w, m_h, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
        }
        m_glow = g;
    }

    QFile df(dir + "/depth.f32");
    if (!df.open(QIODevice::ReadOnly)) return fail("打不开 depth.f32");
    const qint64 need = qint64(m_w) * m_h * sizeof(float);
    if (df.size() != need) return fail(QString("depth.f32 尺寸不符 %1 != %2")
                                           .arg(df.size()).arg(need));
    m_depth.resize(m_w * m_h);
    df.read(reinterpret_cast<char *>(m_depth.data()), need);

    // 主体层深度后处理：保留"圆润的纵深"，但抹掉会把网格撕开的锐利台阶。
    //
    // 两难：完全压平（上一版做法）人像像一张贴纸；不处理又会因为眉骨/鼻梁/
    // 头发之间的深度台阶，让同一张脸的不同部位走不同距离，位移一大就撕。
    // 折中（Kopf 那套 LDI 的简化版）：matte 内部只保留低频起伏（大半径掩码盒式
    // 模糊），边界处再做硬台阶——网格跨轮廓时两侧深度相同，拉伸落在不可见区域。
    if (m_hasMatte && m_matte.width() == m_w && m_matte.height() == m_h) {
        std::vector<uchar> mask(size_t(m_w) * m_h);
        for (int y = 0; y < m_h; ++y) {
            const uchar *sr = m_matte.constScanLine(y);
            for (int x = 0; x < m_w; ++x)
                mask[size_t(y) * m_w + x] = sr[x] >= 6 ? 1 : 0;
        }
        std::vector<float> d(m_depth.begin(), m_depth.end());
        maskedBlur(d, m_w, m_h, mask, qMax(6, m_w / 90));   // 半径≈14px@1272 宽

        // 夹到主体距离附近，避免个别离群点把整块拖走
        if (m_zSubject > 0.05f) {
            const float lo = m_zSubject * 0.80f, hi = m_zSubject * 1.30f;
            for (size_t i = 0; i < d.size(); ++i)
                if (mask[i]) d[i] = qBound(lo, d[i], hi);
        }
        m_depth = QVector<float>(d.begin(), d.end());
    }

    // 再把这块平面的深度向 matte 外侧扩 8px：网格三角形跨轮廓时两侧顶点深度
    // 相同 → 位移一致 → 拉伸全部落在 alpha=0 的不可见区域，可见轮廓保持锐利。
    if (m_hasMatte && m_matte.width() == m_w && m_matte.height() == m_h) {
        const int band = 8;
        std::vector<uchar> known(size_t(m_w) * m_h);
        for (int y = 0; y < m_h; ++y) {
            const uchar *sr = m_matte.constScanLine(y);
            for (int x = 0; x < m_w; ++x)
                known[size_t(y) * m_w + x] = sr[x] >= 6 ? 1 : 0;
        }
        std::vector<float> dz(m_depth.begin(), m_depth.end());
        for (int it = 0; it < band; ++it) {
            bool changed = false;
            std::vector<uchar> next = known;
            for (int y = 1; y < m_h - 1; ++y) {
                for (int x = 1; x < m_w - 1; ++x) {
                    const size_t i = size_t(y) * m_w + x;
                    if (known[i]) continue;
                    const size_t nb[4] = {i - 1, i + 1, i - size_t(m_w), i + size_t(m_w)};
                    for (size_t n : nb) {
                        if (known[n]) { dz[i] = dz[n]; next[i] = 1; changed = true; break; }
                    }
                }
            }
            known.swap(next);
            if (!changed) break;
        }
        m_depth = QVector<float>(dz.begin(), dz.end());
    }

    m_ok = true;
    if (m_gl) {
        // 只重建纹理。**不能**再调 initializeGL()：那会往已经链接好的
        // QOpenGLShaderProgram 里再加一份顶点/片元着色器，链接报
        // "function `main' is multiply defined"，整个程序失效 ——
        // 表现就是"看第二个素材时黑屏"（实测复现）。
        makeCurrent();
        uploadTextures();
        doneCurrent();
    }
    update();
    return true;
}

// 掩码内的盒式模糊（积分图实现，O(1)/像素）。只对 mask=1 的像素取邻域均值，
// 邻域里没有有效像素时保持原值。用来把主体深度压成"只有低频起伏"。
static void maskedBlur(std::vector<float> &d, int w, int h,
                       const std::vector<uchar> &mask, int r)
{
    if (w <= 0 || h <= 0 || r <= 0) return;
    const size_t stride = size_t(w) + 1;
    std::vector<double> si((size_t(w) + 1) * (h + 1), 0.0);
    std::vector<double> sw((size_t(w) + 1) * (h + 1), 0.0);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t i = size_t(y) * w + x;
            const double m = mask[i] ? 1.0 : 0.0;
            const size_t p = size_t(y + 1) * stride + (x + 1);
            si[p] = si[p - 1] + si[p - stride] - si[p - stride - 1] + d[i] * m;
            sw[p] = sw[p - 1] + sw[p - stride] - sw[p - stride - 1] + m;
        }
    }
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t i = size_t(y) * w + x;
            if (!mask[i]) continue;
            const int x0 = qMax(0, x - r), x1 = qMin(w - 1, x + r);
            const int y0 = qMax(0, y - r), y1 = qMin(h - 1, y + r);
            const size_t a = size_t(y0) * stride + x0;
            const size_t b = size_t(y0) * stride + (x1 + 1);
            const size_t c = size_t(y1 + 1) * stride + x0;
            const size_t e = size_t(y1 + 1) * stride + (x1 + 1);
            const double s = si[e] - si[b] - si[c] + si[a];
            const double cnt = sw[e] - sw[b] - sw[c] + sw[a];
            if (cnt > 0.5) d[i] = float(s / cnt);
        }
    }
}

void DibrWidget::initializeGL()
{
    initializeOpenGLFunctions();
    m_gl = true;

    // 着色器和网格只建一次。注意这里**不能**依赖 m_ok：观看页是先 show 再
    // loadAsset，首次 initializeGL 时还没有资产；以前 `if (!m_ok) return;`
    // 会让程序永远没链接着色器。
    if (!m_prog.isLinked()) {
        if (!m_prog.addShaderFromSourceCode(QOpenGLShader::Vertex, kVert))
            qWarning() << "VS:" << m_prog.log();
        if (!m_prog.addShaderFromSourceCode(QOpenGLShader::Fragment, kFrag))
            qWarning() << "FS:" << m_prog.log();
        m_prog.bindAttributeLocation("aPos", 0);
        m_prog.bindAttributeLocation("aUV", 1);
        if (!m_prog.link()) qWarning() << "link:" << m_prog.log();
    }
    if (!m_vao.isCreated()) buildMesh();

    uploadTextures();
}

void DibrWidget::uploadTextures()
{
    if (!m_gl || !m_ok) return;

    GLuint old[7] = {m_texColor, m_texDepth, m_texBlur,
                     m_texMatte, m_texHead, m_texBlink, m_texEye};
    GLuint oldGlow = m_texGlow;
    m_texColor = m_texDepth = m_texBlur = 0;
    m_texMatte = m_texHead = m_texBlink = m_texEye = m_texGlow = 0;

    for (int i = 0; i < 7; ++i)
        if (old[i]) glDeleteTextures(1, &old[i]);
    if (oldGlow) glDeleteTextures(1, &oldGlow);

    glGenTextures(1, &m_texColor);
    glBindTexture(GL_TEXTURE_2D, m_texColor);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_w, m_h, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                 m_img.constBits());

    glGenTextures(1, &m_texBlur);
    glBindTexture(GL_TEXTURE_2D, m_texBlur);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_w, m_h, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                 m_bg.constBits());

    if (m_hasMatte) {
        glGenTextures(1, &m_texMatte);
        glBindTexture(GL_TEXTURE_2D, m_texMatte);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, m_w, m_h, 0, GL_RED,
                     GL_UNSIGNED_BYTE, m_matte.constBits());
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    }

    if (m_hasHead && !m_headMask.isNull()) {
        glGenTextures(1, &m_texHead);
        glBindTexture(GL_TEXTURE_2D, m_texHead);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, m_w, m_h, 0, GL_RED,
                     GL_UNSIGNED_BYTE, m_headMask.constBits());
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    }

    if (m_hasBlink && !m_blinkImg.isNull()) {
        glGenTextures(1, &m_texBlink);
        glBindTexture(GL_TEXTURE_2D, m_texBlink);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_w, m_h, 0, GL_RGBA,
                     GL_UNSIGNED_BYTE, m_blinkImg.constBits());
    }

    if (m_hasMatte && !m_glow.isNull()) {
        glGenTextures(1, &m_texGlow);
        glBindTexture(GL_TEXTURE_2D, m_texGlow);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, m_w, m_h, 0, GL_RED,
                     GL_UNSIGNED_BYTE, m_glow.constBits());
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    }

    if (m_hasEyes && !m_eyeMask.isNull()) {
        glGenTextures(1, &m_texEye);
        glBindTexture(GL_TEXTURE_2D, m_texEye);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, m_w, m_h, 0, GL_RED,
                     GL_UNSIGNED_BYTE, m_eyeMask.constBits());
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    }

    glGenTextures(1, &m_texDepth);
    glBindTexture(GL_TEXTURE_2D, m_texDepth);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, m_w, m_h, 0, GL_RED, GL_FLOAT,
                 m_depth.constData());
}

void DibrWidget::buildMesh()
{
    QVector<float> verts;
    verts.reserve((m_cols + 1) * (m_rows + 1) * 4);
    for (int r = 0; r <= m_rows; ++r) {
        float v = float(r) / m_rows;
        float y = (0.5f - v) * m_h;
        for (int c = 0; c <= m_cols; ++c) {
            float u = float(c) / m_cols;
            float x = (u - 0.5f) * m_w;
            verts << x << y << u << v;
        }
    }

    QVector<unsigned int> idx;
    idx.reserve(m_cols * m_rows * 6);
    const int stride = m_cols + 1;
    for (int r = 0; r < m_rows; ++r) {
        for (int c = 0; c < m_cols; ++c) {
            unsigned int i0 = r * stride + c;
            unsigned int i1 = i0 + 1;
            unsigned int i2 = i0 + stride;
            unsigned int i3 = i2 + 1;
            idx << i0 << i1 << i2 << i1 << i3 << i2;
        }
    }
    m_indexCount = idx.size();

    m_vao.create();
    m_vao.bind();

    m_vbo.create();
    m_vbo.bind();
    m_vbo.allocate(verts.constData(), verts.size() * sizeof(float));

    m_ibo.create();
    m_ibo.bind();
    m_ibo.allocate(idx.constData(), idx.size() * sizeof(unsigned int));

    m_prog.bind();
    m_prog.enableAttributeArray(0);
    m_prog.setAttributeBuffer(0, GL_FLOAT, 0, 2, 4 * sizeof(float));
    m_prog.enableAttributeArray(1);
    m_prog.setAttributeBuffer(1, GL_FLOAT, 2 * sizeof(float), 2, 4 * sizeof(float));
    m_prog.release();

    m_vao.release();
    m_vbo.release();
    m_ibo.release();
}

void DibrWidget::resizeGL(int w, int h)
{
    if (m_w <= 0 || m_h <= 0 || w <= 0 || h <= 0) return;
    const float wa = float(w) / float(h);
    const float aa = float(m_w) / float(m_h);
    if (wa > aa) { m_fitX = aa / wa; m_fitY = 1.0f; }
    else { m_fitX = 1.0f; m_fitY = wa / aa; }
}

void DibrWidget::drawPass(float gainScale, int pass)
{
    m_prog.setUniformValue("uPass", pass);
    m_prog.setUniformValue("uGain", m_gain * gainScale * m_resScale);
    m_vao.bind();
    glDrawElements(GL_TRIANGLES, m_indexCount, GL_UNSIGNED_INT, nullptr);
    m_vao.release();
}

void DibrWidget::paintGL()
{
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    if (!m_ok || !m_texColor || !m_texDepth) return;

    m_time = m_clock.elapsed() / 1000.0f;

    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);

    m_prog.bind();
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_texColor);
    m_prog.setUniformValue("uTex", 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, m_texDepth);
    m_prog.setUniformValue("uDepthTex", 1);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, m_texBlur);
    m_prog.setUniformValue("uTexBlur", 2);
    if (m_hasMatte && m_texMatte) {
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, m_texMatte);
        m_prog.setUniformValue("uMatte", 3);
    }
    if (m_hasMatte && m_texGlow) {
        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_2D, m_texGlow);
        m_prog.setUniformValue("uGlow", 4);
    }
    m_prog.setUniformValue("uHasMatte", m_hasMatte ? 1 : 0);
    if (m_hasBlink && m_texBlink) {
        glActiveTexture(GL_TEXTURE5);
        glBindTexture(GL_TEXTURE_2D, m_texBlink);
        m_prog.setUniformValue("uTexBlink", 5);
    }
    if (m_hasHead && m_texHead) {
        glActiveTexture(GL_TEXTURE6);
        glBindTexture(GL_TEXTURE_2D, m_texHead);
        m_prog.setUniformValue("uHeadMask", 6);
    }
    if (m_hasEyes && m_texEye) {
        glActiveTexture(GL_TEXTURE7);
        glBindTexture(GL_TEXTURE_2D, m_texEye);
        m_prog.setUniformValue("uEyeMask", 7);
        m_prog.setUniformValue("uHasEye", 1.0f);
        // 眼神微动：两条不同频率的正弦叠加，避免看出周期；幅度 ~1.6px
        const float t = m_time;
        const float sx = 1.35f * std::sin(t * 0.41f + 0.7f)
                       + 0.45f * std::sin(t * 1.37f + 2.1f);
        const float sy = 0.75f * std::sin(t * 0.57f + 1.3f)
                       + 0.25f * std::sin(t * 1.71f);
        m_prog.setUniformValue("uEyeShift",
                               QVector2D(sx / float(m_w), sy / float(m_h)));
    } else {
        m_prog.setUniformValue("uHasEye", 0.0f);
        m_prog.setUniformValue("uEyeShift", QVector2D(0.0f, 0.0f));
    }
    m_prog.setUniformValue("uGaze", m_hasHead ? m_gaze : 0.0f);
    m_prog.setUniformValue("uGazeGain", 5.0f);
    m_prog.setUniformValue("uHasBlink", m_hasBlink ? 1 : 0);
    m_prog.setUniformValue("uBlink", m_blink);
    m_prog.setUniformValue("uGhostAlpha", m_ghostAlpha);
    m_prog.setUniformValue("uGlowAmt", m_glowAmt);
    m_prog.setUniformValue("uGrade", m_grade);

    m_prog.setUniformValue("uZFoc", m_zFocus);
    m_prog.setUniformValue("uEyeX", m_eyeX);
    m_prog.setUniformValue("uEyeY", m_eyeY);
    m_prog.setUniformValue("uHalfW", float(m_w) * 0.5f);
    m_prog.setUniformValue("uHalfH", float(m_h) * 0.5f);
    m_prog.setUniformValue("uFitX", m_fitX);
    m_prog.setUniformValue("uFitY", m_fitY);
    m_prog.setUniformValue("uTime", m_time);
    // 视差增益按资产宽度归一。gain=75 是在 1272 px 宽的现场采集资产上标定的
    // （±1 视点在 1.0 m 处约 32 px ≈ 2.5% 画面宽）。换成 180 px 的照片时若照搬
    // 绝对像素，同样的位移会占满画面、边缘露出大片空洞。
    m_resScale = float(m_w) / 1272.0f;
    m_prog.setUniformValue("uPx", m_resScale);
    m_prog.setUniformValue("uMaxShift", 70.0f * m_resScale);
    m_prog.setUniformValue("uSplit", m_split);

    // 背景层：增益 0（不位移），保证铺满整屏不露黑边
    drawPass(0.25f, 0);   // 背景也走一点视差：原来是 0，画面"太死板"

    // 前景层：全增益视差 + matte alpha
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    drawPass(1.0f, 1);

    // 边缘辉光：叠加混合
    if (false) {
        glBlendFunc(GL_ONE, GL_ONE);
        drawPass(0.0f, 2);
    }
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_BLEND);

    m_prog.release();

    ++m_fpsFrames;
    if (m_fpsTimer.elapsed() > 500) {
        m_fps = m_fpsFrames * 1000.0f / m_fpsTimer.elapsed();
        m_fpsFrames = 0;
        m_fpsTimer.restart();
        emit statsUpdated(m_fps, m_eyeX, m_eyeY);
    }
}

QImage DibrWidget::snapshotFrame()
{
    makeCurrent();
    paintGL();
    QImage img = grabFramebuffer();
    doneCurrent();
    return img;
}
