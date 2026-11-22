#pragma once

#include <QElapsedTimer>
#include <QImage>
#include <QOpenGLBuffer>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLWidget>
#include <QString>
#include <QVector>

/**
 * 2.5D DIBR 视差渲染控件。
 *
 * 资产格式（由 tools/make_asset.py 生成）：
 *   rgb.png     校正裁切后的左目图
 *   depth.f32   float32 小端，W*H，单位米
 *   meta.json   {width,height,z_focus,z_min,z_max,...}
 *
 * 渲染：网格位移 + 双层合成
 *   pass 0  背景层：低增益视差，全屏铺底
 *   pass 1  前景层：全增益视差，按深度阈值淡入，压掉前景拉伸
 */
class DibrWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
    Q_OBJECT
public:
    explicit DibrWidget(QWidget *parent = nullptr);

    bool loadAsset(const QString &dir, QString *err = nullptr);
    bool ready() const { return m_ok; }

    void setEye(float ex, float ey);      ///< 归一化视点，-1..1
    void setGain(float g) { m_gain = qBound(0.0f, g, 300.0f); }
    void setFocusDepth(float z) { m_zFocus = qBound(0.2f, z, 30.0f); }
    void setSplit(float z) { m_split = qBound(0.2f, z, 30.0f); }

    float gain() const { return m_gain; }
    float focus() const { return m_zFocus; }
    float split() const { return m_split; }
    int assetWidth() const { return m_w; }
    int assetHeight() const { return m_h; }
    float fps() const { return m_fps; }

    /// 真实渲染一帧并取回（用于离屏验证）
    QImage snapshotFrame();

signals:
    void statsUpdated(float fps, float eyeX, float eyeY);

protected:
    void initializeGL() override;
    void paintGL() override;
    void resizeGL(int w, int h) override;

private:
    void buildMesh();
    void drawPass(float gainScale, int pass);

    bool m_ok = false;
    bool m_gl = false;
    int m_w = 0, m_h = 0;
    QImage m_img;
    QImage m_blur;
    QImage m_bg;
    QVector<float> m_depth;
    float m_zFocus = 5.0f, m_zMin = 0.3f, m_zMax = 10.0f;
    float m_split = 2.0f;
    float m_eyeX = 0.0f, m_eyeY = 0.0f;
    float m_gain = 40.0f;
    float m_fps = 0.0f;
    float m_fitX = 1.0f, m_fitY = 1.0f;

    QOpenGLShaderProgram m_prog;
    QOpenGLBuffer m_vbo{QOpenGLBuffer::VertexBuffer};
    QOpenGLBuffer m_ibo{QOpenGLBuffer::IndexBuffer};
    QOpenGLVertexArrayObject m_vao;
    GLuint m_texColor = 0, m_texDepth = 0, m_texBlur = 0;
    int m_cols = 220, m_rows = 124;
    int m_indexCount = 0;

    QElapsedTimer m_clock;
    float m_time = 0.0f;
    QElapsedTimer m_fpsTimer;
    int m_fpsFrames = 0;
};
