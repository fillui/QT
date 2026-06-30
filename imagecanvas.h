#ifndef IMAGECANVAS_H
#define IMAGECANVAS_H

#include <QWidget>
#include <QImage>
#include <QPoint>
#include <QPointF>
#include <QRect>
#include <QVector>
#include <QString>
#include <QTransform>
#include <QList>
#include <QPair>
#include <QPolygonF>

#include "annotation.h"

class ImageCanvas : public QWidget
{
    Q_OBJECT

public:
    explicit ImageCanvas(QWidget *parent = nullptr);

    void loadImage(const QString &imagePath);
    void setImage(const QString &imagePath);
    void clearImage();

    void setCurrentLabel(const QString &label);
    void clearAnnotations(bool recordUndo = true);
    void setAnnotations(const QVector<Annotation> &newAnnotations);
    void deleteSelectedAnnotation();
    void selectAnnotation(int index);
    void changeSelectedAnnotationLabel(const QString &label);

    void undo();
    void redo();
    bool canUndo() const;
    bool canRedo() const;

    // 图像常见操作：放大、缩小、平移、旋转、复位
    void zoomIn();
    void zoomOut();
    void resetView();
    void rotateLeft();
    void rotateRight();

    QString currentImagePath() const;

    // 为了和 mainwindow.cpp 直接配合，这两个保留为 public
    QVector<Annotation> annotations;
    int selectedIndex = -1;

signals:
    void annotationsChanged();

protected:
    void paintEvent(QPaintEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
    void leaveEvent(QEvent *event) override;

private:
    enum class DragMode {
        None,
        Drawing,
        Moving,
        Resizing,
        Panning
    };

    enum class ResizeHandle {
        None,
        Left,
        Right,
        Top,
        Bottom,
        TopLeft,
        TopRight,
        BottomLeft,
        BottomRight
    };

    QTransform imageToViewTransform() const;
    QTransform viewToImageTransform() const;
    QRect imageDrawRect() const;
    QRect imageRectToViewRect(const QRect &imageRect) const;
    QPolygonF imageRectToViewPolygon(const QRect &imageRect) const;
    QPoint viewPointToImagePoint(const QPoint &viewPoint) const;
    bool viewPointInsideImage(const QPoint &viewPoint) const;
    QRect clampRectToImage(const QRect &imageRect) const;
    QRect movedRectInImage(const QRect &rect, const QPoint &delta) const;

    QList<QPair<ResizeHandle, QPointF>> resizeHandlePoints(const QRect &imageRect) const;
    ResizeHandle hitResizeHandle(const QRect &imageRect, const QPoint &viewPoint) const;
    int hitAnnotation(const QPoint &viewPoint, ResizeHandle *handle = nullptr) const;
    void updateHoverCursor(const QPoint &viewPoint);
    void pushUndoState();
    void restoreAnnotations(const QVector<Annotation> &state);
    void applyZoom(double factor, const QPointF &anchorViewPoint);

private:
    QImage m_image;
    QString m_imagePath;

    DragMode m_dragMode = DragMode::None;
    ResizeHandle m_resizeHandle = ResizeHandle::None;

    bool m_dragChanged = false;
    QPoint m_startImagePoint;
    QPoint m_lastImagePoint;
    QRect m_originalRect;
    QRect m_tempImageRect;

    QPoint m_panStartViewPoint;
    QPointF m_panStartOffset;
    QPointF m_panOffset;
    double m_zoomFactor = 1.0;
    int m_rotationDegrees = 0;

    QString m_currentLabel = "object";

    QVector<QVector<Annotation>> m_undoStack;
    QVector<QVector<Annotation>> m_redoStack;
};

#endif // IMAGECANVAS_H
