#include "annotationio.h"

#include <QFileInfo>
#include <QSaveFile>
#include <QFile>
#include <QTextStream>
#include <QRegularExpression>

namespace AnnotationIO
{
QString labelFilePath(const QString& imagePath)
{
    QFileInfo info(imagePath);
    return info.absolutePath() + "/" + info.completeBaseName() + ".txt";
}

bool save(const QString& imagePath,
          const QVector<Annotation>& annotations,
          QString* errorMessage)
{
    const QString filePath = labelFilePath(imagePath);

    QSaveFile file(filePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        if (errorMessage)
            *errorMessage = QString("无法写入标注文件: %1").arg(filePath);
        return false;
    }

    QTextStream out(&file);
    for (const auto& ann : annotations)
    {
        QString label = ann.label.simplified();
        label.replace(' ', '_');

        out << label << " "
            << ann.rect.left() << " "
            << ann.rect.top() << " "
            << ann.rect.right() << " "
            << ann.rect.bottom() << "\n";
    }

    if (!file.commit())
    {
        if (errorMessage)
            *errorMessage = QString("保存失败: %1").arg(filePath);
        return false;
    }

    return true;
}

bool load(const QString& imagePath,
          QVector<Annotation>& annotations,
          QString* errorMessage)
{
    annotations.clear();
    const QString filePath = labelFilePath(imagePath);

    QFile file(filePath);
    if (!file.exists())
        return true;

    if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
    {
        if (errorMessage)
            *errorMessage = QString("无法读取标注文件: %1").arg(filePath);
        return false;
    }

    QTextStream in(&file);

    while (!in.atEnd())
    {
        QString line = in.readLine().trimmed();
        if (line.isEmpty())
            continue;

        QStringList parts = line.split(QRegularExpression("\\s+"),
                                       Qt::SkipEmptyParts);
        if (parts.size() < 5)
            continue;

        bool ok1, ok2, ok3, ok4;
        int x1 = parts[1].toInt(&ok1);
        int y1 = parts[2].toInt(&ok2);
        int x2 = parts[3].toInt(&ok3);
        int y2 = parts[4].toInt(&ok4);

        if (!(ok1 && ok2 && ok3 && ok4))
            continue;

        Annotation ann;
        ann.label = parts[0];
        ann.rect = QRect(QPoint(x1, y1), QPoint(x2, y2)).normalized();
        annotations.push_back(ann);
    }

    return true;
}
}