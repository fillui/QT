#ifndef ANNOTATION_H
#define ANNOTATION_H

#include <QRect>
#include <QString>

struct Annotation
{
    QRect rect;      // 原图坐标
    QString label;   // 类别名
};

#endif // ANNOTATION_H
