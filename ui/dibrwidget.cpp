#include "dibrwidget.h"

#include <QDebug>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>

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
uniform float uTime, uMaxShift;

varying vec2 vUV;

void main() {
    float z = texture2D(uDepthTex, aUV).r;
    z = max(z, 0.15);

    // 视差与 1/z 之差成正比：焦点处为 0，更近的前移、更远的后移
    float p = (1.0 / uZFoc) - (1.0 / z);
    vec2 disp = vec2(uEyeX, uEyeY) * uGain * p;
    disp = clamp(disp, vec2(-uMaxShift), vec2(uMaxShift));

    vec2 pos = aPos + disp;

    // 呼吸：以画面中心为基准的极轻微纵向缩放 + 横向漂移
    pos.y *= 1.0 + 0.0026 * sin(uTime * 1.6);
    pos.x += 0.8 * sin(uTime * 0.7);

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
        vec3 c = texture2D(uTex, vUV).rgb;
        if (uHasBlink == 1 && uBlink > 0.001) {
            vec3 c2 = texture2D(uTexBlink, vUV).rgb;
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
        float rim = clamp(g - m, 0.0, 1.0);
        rim = pow(rim, 0.6) * uGlowAmt * 2.2;   // 提亮，让辉光真的可见
        vec3 warm = vec3(1.0, 0.87, 0.70);
        gl_FragColor = vec4(warm * rim, 1.0);
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

    if (m_hasMatte) {
        // 辉光贴图 = 模糊后的 matte；着色器里用 (glow - matte) 得到边缘带
        m_glow = m_matte.scaled(qMax(8, m_w / 12), qMax(8, m_h / 12),
                                Qt::IgnoreAspectRatio, Qt::SmoothTransformation)
                        .scaled(m_w, m_h, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
    }

    QFile df(dir + "/depth.f32");
    if (!df.open(QIODevice::ReadOnly)) return fail("打不开 depth.f32");
    const qint64 need = qint64(m_w) * m_h * sizeof(float);
    if (df.size() != need) return fail(QString("depth.f32 尺寸不符 %1 != %2")
                                           .arg(df.size()).arg(need));
    m_depth.resize(m_w * m_h);
    df.read(reinterpret_cast<char *>(m_depth.data()), need);

    m_ok = true;
    if (m_gl) {
        makeCurrent();
        if (m_texColor) {
            glDeleteTextures(1, &m_texColor);
            glDeleteTextures(1, &m_texDepth);
            glDeleteTextures(1, &m_texBlur);
        }
        m_texColor = m_texDepth = m_texBlur = 0;
        initializeGL();
        doneCurrent();
    }
    update();
    return true;
}

void DibrWidget::initializeGL()
{
    initializeOpenGLFunctions();
    m_gl = true;
    if (!m_ok) return;

    if (!m_prog.addShaderFromSourceCode(QOpenGLShader::Vertex, kVert))
        qWarning() << "VS:" << m_prog.log();
    if (!m_prog.addShaderFromSourceCode(QOpenGLShader::Fragment, kFrag))
        qWarning() << "FS:" << m_prog.log();
    m_prog.bindAttributeLocation("aPos", 0);
    m_prog.bindAttributeLocation("aUV", 1);
    if (!m_prog.link()) qWarning() << "link:" << m_prog.log();

    buildMesh();

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
    m_prog.setUniformValue("uGain", m_gain * gainScale);
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
    m_prog.setUniformValue("uMaxShift", 70.0f);
    m_prog.setUniformValue("uSplit", m_split);

    // 背景层：增益 0（不位移），保证铺满整屏不露黑边
    drawPass(0.0f, 0);

    // 前景层：全增益视差 + matte alpha
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    drawPass(1.0f, 1);

    // 边缘辉光：叠加混合
    if (m_hasMatte && m_texGlow && m_glowAmt > 0.01f) {
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
