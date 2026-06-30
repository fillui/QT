#ifndef ANNOTATIONIO_H
#define ANNOTATIONIO_H

#include "annotation.h"
#include <QString>
#include <QVector>

namespace AnnotationIO
{
QString labelFilePath(const QString& imagePath);

bool save(const QString& imagePath,
          const QVector<Annotation>& annotations,
          QString* errorMessage = nullptr);

bool load(const QString& imagePath,
          QVector<Annotation>& annotations,
          QString* errorMessage = nullptr);
}

#endif // ANNOTATIONIO_H