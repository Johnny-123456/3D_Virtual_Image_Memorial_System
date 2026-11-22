#pragma once

#include <QElapsedTimer>
#include <QObject>
#include <QString>
#include <QUdpSocket>

/**
 * 接收眼位追踪器的 UDP 数据。
 *
 * 协议：127.0.0.1:<port>，一行 ASCII
 *   "ex,ey\n"   归一化视点，各 -1..1
 *   "nan,nan"   检测不到人，保持上一次位置
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
            if (xy.size() != 2) continue;
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
};
