#pragma once

#include <QDateTime>
#include <QDebug>
#include <QElapsedTimer>
#include <QObject>
#include <QString>
#include <QUdpSocket>

/**
 * 接收眼位追踪器的 UDP 数据。
 *
 * 协议：127.0.0.1:<port>，一行 ASCII
 *   "ex,ey\n"        归一化视点，各 -1..1
 *   "nan,nan\n"      检测不到人，保持上一次位置
 *   "ex,ey,<epoch>"  追踪器带 --stamp 时多一个抓帧时刻（秒），用于量端到端延迟
 */
class EyeLink : public QObject
{
    Q_OBJECT
public:
    explicit EyeLink(QObject *parent = nullptr) : QObject(parent) { m_last.start(); }

    bool start(quint16 port, QString *err = nullptr)
    {
        if (!m_sock.bind(QHostAddress::LocalHost, port,
                         QUdpSocket::ShareAddress | QUdpSocket::ReuseAddressHint)) {
            if (err) *err = m_sock.errorString();
            return false;
        }
        connect(&m_sock, &QUdpSocket::readyRead, this, &EyeLink::onReadyRead);
        m_last.start();
        return true;
    }

    /// 最近 1.5 秒内是否收到过有效数据
    bool active() const { return m_last.isValid() && m_last.elapsed() < 1500; }

    /// 最近一次收到的抓帧时刻（epoch 秒，0 = 没有时间戳）
    double lastStamp() const { return m_stamp; }

signals:
    void eye(float ex, float ey);
    void lost();

private slots:
    void onReadyRead()
    {
        while (m_sock.hasPendingDatagrams()) {
            QByteArray buf(int(m_sock.pendingDatagramSize()), '\0');
            m_sock.readDatagram(buf.data(), buf.size());
            const QString s = QString::fromLatin1(buf).trimmed();
            const QStringList xy = s.split(',');

            // 延迟统计放在坐标判定之前：检测不到人时追踪器照样发包，
            // 管线耗时一样，仍然值得量。
            if (xy.size() >= 3) {
                const double st = xy[2].toDouble();
                if (st > 0.0) {
                    const double now = QDateTime::currentMSecsSinceEpoch() / 1000.0;
                    const double ms = (now - st) * 1000.0;
                    m_latEma = m_latN == 0 ? ms : (0.8 * m_latEma + 0.2 * ms);
                    if (++m_latN % 30 == 0)
                        qInfo("latency avg=%.1f ms (n=%d)", m_latEma, m_latN);
                    m_stamp = st;
                }
            }

            if (xy.size() < 2) continue;
            bool ok1 = false, ok2 = false;
            const float ex = xy[0].toFloat(&ok1);
            const float ey = xy[1].toFloat(&ok2);
            if (!ok1 || !ok2) {
                emit lost();      // nan,nan
                continue;
            }
            m_last.restart();
            emit eye(ex, ey);
        }
    }

private:
    QUdpSocket m_sock;
    QElapsedTimer m_last;
    double m_stamp = 0.0;
    double m_latEma = 0.0;
    int m_latN = 0;
};
