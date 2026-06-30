#include "imagecanvas.h"

#include <QColor>
#include <QEvent>
#include <QMouseEvent>
#include <QPainter>
#include <QPen>
#include <QWheelEvent>
#include <QtMath>

namespace {
constexpr int kMinBoxSize = 5;
constexpr int kHandleSize = 8;
constexpr int kMaxUndoDepth = 50;
constexpr double kMinZoom = 0.2;
constexpr double kMaxZoom = 8.0;
}

ImageCanvas::ImageCanvas(QWidget *parent)
    : QWidget(parent)
{
    setMouseTracking(true);
    setMinimumSize(600, 400);
    setFocusPolicy(Qt::StrongFocus);
}

void ImageCanvas::loadImage(const QString &imagePath)
{
    QImage img;

    if (!img.load(imagePath)) {
        clearImage();
        return;
    }

    m_image = img;
    m_imagePath = imagePath;
    m_dragMode = DragMode::None;
    m_dragChanged = false;
    selectedIndex = -1;

    // 换图时恢复默认视图，避免上一张图片的缩放/平移影响新图片。
    m_zoomFactor = 1.0;
    m_panOffset = QPointF(0, 0);
    m_rotationDegrees = 0;

    update();
}

void ImageCanvas::setImage(const QString &imagePath)
{
    loadImage(imagePath);
}

void ImageCanvas::clearImage()
{
    m_image = QImage();
    m_imagePath.clear();
    annotations.clear();
    selectedIndex = -1;
    m_dragMode = DragMode::None;
    m_dragChanged = false;
    m_undoStack.clear();
    m_redoStack.clear();
    m_zoomFactor = 1.0;
    m_panOffset = QPointF(0, 0);
    m_rotationDegrees = 0;

    unsetCursor();
    update();
    emit annotationsChanged();
}

void ImageCanvas::setCurrentLabel(const QString &label)
{
    const QString t = label.trimmed();

    if (!t.isEmpty()) {
        m_currentLabel = t;
    }
}

void ImageCanvas::clearAnnotations(bool recordUndo)
{
    if (annotations.isEmpty()) {
        selectedIndex = -1;
        update();
        emit annotationsChanged();
        return;
    }

    if (recordUndo) {
        pushUndoState();
    }

    annotations.clear();
    selectedIndex = -1;
    m_dragMode = DragMode::None;
    m_dragChanged = false;

    update();
    emit annotationsChanged();
}

void ImageCanvas::setAnnotations(const QVector<Annotation> &newAnnotations)
{
    annotations = newAnnotations;
    selectedIndex = annotations.isEmpty() ? -1 : 0;
    m_dragMode = DragMode::None;
    m_dragChanged = false;
    m_undoStack.clear();
    m_redoStack.clear();

    update();
    emit annotationsChanged();
}

void ImageCanvas::deleteSelectedAnnotation()
{
    if (selectedIndex < 0 || selectedIndex >= annotations.size()) {
        return;
    }

    pushUndoState();
    annotations.removeAt(selectedIndex);

    if (annotations.isEmpty()) {
        selectedIndex = -1;
    } else if (selectedIndex >= annotations.size()) {
        selectedIndex = annotations.size() - 1;
    }

    update();
    emit annotationsChanged();
}

void ImageCanvas::selectAnnotation(int index)
{
    if (index >= 0 && index < annotations.size()) {
        selectedIndex = index;
    } else {
        selectedIndex = -1;
    }

    update();
}

void ImageCanvas::changeSelectedAnnotationLabel(const QString &label)
{
    const QString t = label.trimmed();

    if (t.isEmpty()) {
        return;
    }

    if (selectedIndex < 0 || selectedIndex >= annotations.size()) {
        return;
    }

    if (annotations[selectedIndex].label == t) {
        return;
    }

    pushUndoState();
    annotations[selectedIndex].label = t;

    update();
    emit annotationsChanged();
}

void ImageCanvas::undo()
{
    if (m_undoStack.isEmpty()) {
        return;
    }

    m_redoStack.append(annotations);
    QVector<Annotation> state = m_undoStack.takeLast();
    restoreAnnotations(state);
}

void ImageCanvas::redo()
{
    if (m_redoStack.isEmpty()) {
        return;
    }

    m_undoStack.append(annotations);
    QVector<Annotation> state = m_redoStack.takeLast();
    restoreAnnotations(state);
}

bool ImageCanvas::canUndo() const
{
    return !m_undoStack.isEmpty();
}

bool ImageCanvas::canRedo() const
{
    return !m_redoStack.isEmpty();
}

void ImageCanvas::zoomIn()
{
    applyZoom(1.25, QPointF(width() / 2.0, height() / 2.0));
}

void ImageCanvas::zoomOut()
{
    applyZoom(0.8, QPointF(width() / 2.0, height() / 2.0));
}

void ImageCanvas::resetView()
{
    m_zoomFactor = 1.0;
    m_panOffset = QPointF(0, 0);
    m_rotationDegrees = 0;
    update();
}

void ImageCanvas::rotateLeft()
{
    m_rotationDegrees = (m_rotationDegrees + 270) % 360;
    update();
}

void ImageCanvas::rotateRight()
{
    m_rotationDegrees = (m_rotationDegrees + 90) % 360;
    update();
}

QString ImageCanvas::currentImagePath() const
{
    return m_imagePath;
}

QTransform ImageCanvas::imageToViewTransform() const
{
    QTransform transform;

    if (m_image.isNull() || width() <= 0 || height() <= 0) {
        return transform;
    }

    const double iw = m_image.width();
    const double ih = m_image.height();

    QTransform rotateTransform;
    rotateTransform.translate(iw / 2.0, ih / 2.0);
    rotateTransform.rotate(m_rotationDegrees);
    rotateTransform.translate(-iw / 2.0, -ih / 2.0);

    const QRectF rotatedBounds = rotateTransform.mapRect(QRectF(0, 0, iw, ih));
    const double fitScale = qMin(static_cast<double>(width()) / rotatedBounds.width(),
                                 static_cast<double>(height()) / rotatedBounds.height());
    const double scale = fitScale * m_zoomFactor;

    const double targetW = rotatedBounds.width() * scale;
    const double targetH = rotatedBounds.height() * scale;
    const double left = (width() - targetW) / 2.0 + m_panOffset.x();
    const double top = (height() - targetH) / 2.0 + m_panOffset.y();

    transform.translate(left, top);
    transform.scale(scale, scale);
    transform.translate(-rotatedBounds.left(), -rotatedBounds.top());
    transform.translate(iw / 2.0, ih / 2.0);
    transform.rotate(m_rotationDegrees);
    transform.translate(-iw / 2.0, -ih / 2.0);

    return transform;
}

QTransform ImageCanvas::viewToImageTransform() const
{
    bool invertible = false;
    const QTransform inv = imageToViewTransform().inverted(&invertible);
    return invertible ? inv : QTransform();
}

QRect ImageCanvas::imageDrawRect() const
{
    if (m_image.isNull()) {
        return QRect();
    }

    const QPolygonF poly = imageToViewTransform().map(QPolygonF(QRectF(0, 0, m_image.width(), m_image.height())));
    return poly.boundingRect().toAlignedRect();
}

QRect ImageCanvas::imageRectToViewRect(const QRect &imageRect) const
{
    if (m_image.isNull()) {
        return QRect();
    }

    return imageRectToViewPolygon(imageRect).boundingRect().toAlignedRect();
}

QPolygonF ImageCanvas::imageRectToViewPolygon(const QRect &imageRect) const
{
    if (m_image.isNull()) {
        return QPolygonF();
    }

    QRectF r = QRectF(imageRect.normalized());
    return imageToViewTransform().map(QPolygonF(r));
}

QPoint ImageCanvas::viewPointToImagePoint(const QPoint &viewPoint) const
{
    if (m_image.isNull()) {
        return QPoint();
    }

    const QPointF p = viewToImageTransform().map(QPointF(viewPoint));

    int x = qRound(p.x());
    int y = qRound(p.y());

    x = qBound(0, x, m_image.width() - 1);
    y = qBound(0, y, m_image.height() - 1);

    return QPoint(x, y);
}

bool ImageCanvas::viewPointInsideImage(const QPoint &viewPoint) const
{
    if (m_image.isNull()) {
        return false;
    }

    const QPointF p = viewToImageTransform().map(QPointF(viewPoint));
    return QRectF(0, 0, m_image.width(), m_image.height()).contains(p);
}

QRect ImageCanvas::clampRectToImage(const QRect &imageRect) const
{
    if (m_image.isNull()) {
        return QRect();
    }

    QRect r = imageRect.normalized();
    QRect bounds(0, 0, m_image.width(), m_image.height());
    r = r.intersected(bounds);

    if (r.width() < 1 || r.height() < 1) {
        return QRect();
    }

    return r;
}

QRect ImageCanvas::movedRectInImage(const QRect &rect, const QPoint &delta) const
{
    if (m_image.isNull()) {
        return rect;
    }

    QRect r = rect.translated(delta);

    if (r.left() < 0) {
        r.moveLeft(0);
    }

    if (r.top() < 0) {
        r.moveTop(0);
    }

    if (r.right() >= m_image.width()) {
        r.moveRight(m_image.width() - 1);
    }

    if (r.bottom() >= m_image.height()) {
        r.moveBottom(m_image.height() - 1);
    }

    return r;
}

QList<QPair<ImageCanvas::ResizeHandle, QPointF>> ImageCanvas::resizeHandlePoints(const QRect &imageRect) const
{
    QList<QPair<ResizeHandle, QPointF>> result;
    const QRectF r(imageRect.normalized());
    const QTransform t = imageToViewTransform();

    const QPointF tl = t.map(r.topLeft());
    const QPointF tr = t.map(r.topRight());
    const QPointF bl = t.map(r.bottomLeft());
    const QPointF br = t.map(r.bottomRight());
    const QPointF left = t.map(QPointF(r.left(), r.center().y()));
    const QPointF right = t.map(QPointF(r.right(), r.center().y()));
    const QPointF top = t.map(QPointF(r.center().x(), r.top()));
    const QPointF bottom = t.map(QPointF(r.center().x(), r.bottom()));

    result << qMakePair(ResizeHandle::TopLeft, tl)
           << qMakePair(ResizeHandle::TopRight, tr)
           << qMakePair(ResizeHandle::BottomLeft, bl)
           << qMakePair(ResizeHandle::BottomRight, br)
           << qMakePair(ResizeHandle::Left, left)
           << qMakePair(ResizeHandle::Right, right)
           << qMakePair(ResizeHandle::Top, top)
           << qMakePair(ResizeHandle::Bottom, bottom);

    return result;
}

ImageCanvas::ResizeHandle ImageCanvas::hitResizeHandle(const QRect &imageRect, const QPoint &viewPoint) const
{
    const int hs = kHandleSize;

    for (const auto &pair : resizeHandlePoints(imageRect)) {
        const QPointF p = pair.second;
        const QRectF handleRect(p.x() - hs / 2.0, p.y() - hs / 2.0, hs, hs);

        if (handleRect.contains(QPointF(viewPoint))) {
            return pair.first;
        }
    }

    return ResizeHandle::None;
}

int ImageCanvas::hitAnnotation(const QPoint &viewPoint, ResizeHandle *handle) const
{
    if (handle) {
        *handle = ResizeHandle::None;
    }

    if (m_image.isNull() || !viewPointInsideImage(viewPoint)) {
        return -1;
    }

    // 已选中的框优先判断 8 个编辑控制点
    if (selectedIndex >= 0 && selectedIndex < annotations.size()) {
        ResizeHandle h = hitResizeHandle(annotations[selectedIndex].rect, viewPoint);

        if (h != ResizeHandle::None) {
            if (handle) {
                *handle = h;
            }
            return selectedIndex;
        }
    }

    const QPoint imagePoint = viewPointToImagePoint(viewPoint);

    // 倒序命中，优先选择后画的框
    for (int i = annotations.size() - 1; i >= 0; --i) {
        if (annotations[i].rect.normalized().contains(imagePoint)) {
            return i;
        }
    }

    return -1;
}

void ImageCanvas::updateHoverCursor(const QPoint &viewPoint)
{
    if (m_image.isNull()) {
        unsetCursor();
        return;
    }

    ResizeHandle handle = ResizeHandle::None;
    const int index = hitAnnotation(viewPoint, &handle);

    if (handle == ResizeHandle::TopLeft || handle == ResizeHandle::BottomRight) {
        setCursor(Qt::SizeFDiagCursor);
    } else if (handle == ResizeHandle::TopRight || handle == ResizeHandle::BottomLeft) {
        setCursor(Qt::SizeBDiagCursor);
    } else if (handle == ResizeHandle::Left || handle == ResizeHandle::Right) {
        setCursor(Qt::SizeHorCursor);
    } else if (handle == ResizeHandle::Top || handle == ResizeHandle::Bottom) {
        setCursor(Qt::SizeVerCursor);
    } else if (index >= 0) {
        setCursor(Qt::SizeAllCursor);
    } else if (viewPointInsideImage(viewPoint)) {
        setCursor(Qt::CrossCursor);
    } else {
        unsetCursor();
    }
}

void ImageCanvas::pushUndoState()
{
    m_undoStack.append(annotations);

    if (m_undoStack.size() > kMaxUndoDepth) {
        m_undoStack.removeFirst();
    }

    m_redoStack.clear();
}

void ImageCanvas::restoreAnnotations(const QVector<Annotation> &state)
{
    annotations = state;
    selectedIndex = annotations.isEmpty() ? -1 : qMin(selectedIndex, annotations.size() - 1);
    m_dragMode = DragMode::None;
    m_dragChanged = false;

    update();
    emit annotationsChanged();
}

void ImageCanvas::applyZoom(double factor, const QPointF &anchorViewPoint)
{
    if (m_image.isNull()) {
        return;
    }

    const double oldZoom = m_zoomFactor;
    const double newZoom = qBound(kMinZoom, oldZoom * factor, kMaxZoom);

    if (qFuzzyCompare(oldZoom, newZoom)) {
        return;
    }

    // 保持鼠标所在的图像点尽量不漂移。
    const QPointF imagePoint = viewToImageTransform().map(anchorViewPoint);
    m_zoomFactor = newZoom;
    const QPointF newViewPoint = imageToViewTransform().map(imagePoint);
    m_panOffset += anchorViewPoint - newViewPoint;

    update();
}

void ImageCanvas::paintEvent(QPaintEvent *event)
{
    Q_UNUSED(event);

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.fillRect(rect(), QColor(28, 28, 28));

    if (m_image.isNull()) {
        painter.setPen(Qt::white);
        painter.drawText(rect(), Qt::AlignCenter, "请先打开图片文件夹");
        return;
    }

    const QTransform imageTransform = imageToViewTransform();
    const QPolygonF imagePolygon = imageTransform.map(QPolygonF(QRectF(0, 0, m_image.width(), m_image.height())));

    painter.save();
    painter.setTransform(imageTransform);
    painter.drawImage(QPointF(0, 0), m_image);
    painter.restore();

    // 绘制图像边界，旋转后也能看出实际图片区域。
    QPen imageBorder(QColor(80, 80, 80));
    imageBorder.setWidth(1);
    painter.setPen(imageBorder);
    painter.setBrush(Qt::NoBrush);
    painter.drawPolygon(imagePolygon);

    for (int i = 0; i < annotations.size(); ++i) {
        const Annotation &ann = annotations[i];
        const QPolygonF boxPoly = imageRectToViewPolygon(ann.rect);
        const QRect viewRect = boxPoly.boundingRect().toAlignedRect();

        QPen pen(i == selectedIndex ? QColor(255, 200, 0) : QColor(0, 180, 255));
        pen.setWidth(2);
        painter.setPen(pen);
        painter.setBrush(Qt::NoBrush);
        painter.drawPolygon(boxPoly);

        QRect labelRect(viewRect.left(), viewRect.top() - 24, 130, 24);
        if (labelRect.top() < 0) {
            labelRect.moveTop(viewRect.top());
        }

        painter.fillRect(labelRect, i == selectedIndex ? QColor(255, 200, 0) : QColor(0, 180, 255));
        painter.setPen(Qt::black);
        painter.drawText(labelRect.adjusted(6, 0, -4, 0),
                         Qt::AlignVCenter | Qt::AlignLeft,
                         ann.label);

        // 选中框显示 8 个编辑控制点，用于拖动缩放。
        if (i == selectedIndex) {
            painter.setPen(Qt::NoPen);
            painter.setBrush(QColor(255, 200, 0));

            const auto points = resizeHandlePoints(ann.rect);
            for (const auto &pair : points) {
                const QPointF p = pair.second;
                painter.drawRect(QRectF(p.x() - kHandleSize / 2.0,
                                        p.y() - kHandleSize / 2.0,
                                        kHandleSize,
                                        kHandleSize));
            }
        }
    }

    if (m_dragMode == DragMode::Drawing) {
        const QPolygonF tempPoly = imageRectToViewPolygon(m_tempImageRect);

        QPen pen(QColor(255, 200, 0));
        pen.setWidth(2);
        pen.setStyle(Qt::DashLine);
        painter.setPen(pen);
        painter.setBrush(Qt::NoBrush);
        painter.drawPolygon(tempPoly);
    }
}

void ImageCanvas::mousePressEvent(QMouseEvent *event)
{
    if (m_image.isNull()) {
        return;
    }

    if (event->button() == Qt::RightButton) {
        m_dragMode = DragMode::Panning;
        m_panStartViewPoint = event->pos();
        m_panStartOffset = m_panOffset;
        setCursor(Qt::ClosedHandCursor);
        return;
    }

    if (event->button() != Qt::LeftButton) {
        return;
    }

    if (!viewPointInsideImage(event->pos())) {
        selectedIndex = -1;
        update();
        emit annotationsChanged();
        return;
    }

    ResizeHandle handle = ResizeHandle::None;
    const int index = hitAnnotation(event->pos(), &handle);

    if (index >= 0) {
        selectedIndex = index;
        m_startImagePoint = viewPointToImagePoint(event->pos());
        m_lastImagePoint = m_startImagePoint;
        m_originalRect = annotations[selectedIndex].rect;
        m_resizeHandle = handle;
        m_dragChanged = false;

        if (handle != ResizeHandle::None) {
            m_dragMode = DragMode::Resizing;
        } else {
            m_dragMode = DragMode::Moving;
        }

        pushUndoState();
        updateHoverCursor(event->pos());
        update();
        emit annotationsChanged();
        return;
    }

    selectedIndex = -1;
    m_dragMode = DragMode::Drawing;
    m_resizeHandle = ResizeHandle::None;
    m_startImagePoint = viewPointToImagePoint(event->pos());
    m_lastImagePoint = m_startImagePoint;
    m_tempImageRect = QRect(m_startImagePoint, m_lastImagePoint).normalized();
    m_dragChanged = false;

    update();
    emit annotationsChanged();
}

void ImageCanvas::mouseMoveEvent(QMouseEvent *event)
{
    if (m_image.isNull()) {
        return;
    }

    if (m_dragMode == DragMode::None) {
        updateHoverCursor(event->pos());
        return;
    }

    if (m_dragMode == DragMode::Panning) {
        m_panOffset = m_panStartOffset + (event->pos() - m_panStartViewPoint);
        update();
        return;
    }

    const QPoint current = viewPointToImagePoint(event->pos());

    if (m_dragMode == DragMode::Drawing) {
        m_tempImageRect = clampRectToImage(QRect(m_startImagePoint, current).normalized());
        m_dragChanged = true;
        update();
        return;
    }

    if (selectedIndex < 0 || selectedIndex >= annotations.size()) {
        return;
    }

    if (m_dragMode == DragMode::Moving) {
        const QPoint delta = current - m_startImagePoint;
        QRect moved = movedRectInImage(m_originalRect, delta);
        annotations[selectedIndex].rect = moved;
        m_dragChanged = true;
        update();
        return;
    }

    if (m_dragMode == DragMode::Resizing) {
        QRect r = m_originalRect;

        switch (m_resizeHandle) {
        case ResizeHandle::Left:
            r.setLeft(current.x());
            break;
        case ResizeHandle::Right:
            r.setRight(current.x());
            break;
        case ResizeHandle::Top:
            r.setTop(current.y());
            break;
        case ResizeHandle::Bottom:
            r.setBottom(current.y());
            break;
        case ResizeHandle::TopLeft:
            r.setTopLeft(current);
            break;
        case ResizeHandle::TopRight:
            r.setTopRight(current);
            break;
        case ResizeHandle::BottomLeft:
            r.setBottomLeft(current);
            break;
        case ResizeHandle::BottomRight:
            r.setBottomRight(current);
            break;
        case ResizeHandle::None:
            break;
        }

        r = clampRectToImage(r.normalized());

        if (r.width() >= kMinBoxSize && r.height() >= kMinBoxSize) {
            annotations[selectedIndex].rect = r;
            m_dragChanged = true;
        }

        update();
    }
}

void ImageCanvas::mouseReleaseEvent(QMouseEvent *event)
{
    if (m_image.isNull()) {
        return;
    }

    if (event->button() == Qt::RightButton && m_dragMode == DragMode::Panning) {
        m_dragMode = DragMode::None;
        updateHoverCursor(event->pos());
        return;
    }

    if (event->button() != Qt::LeftButton) {
        return;
    }

    if (m_dragMode == DragMode::Drawing) {
        const QPoint current = viewPointToImagePoint(event->pos());
        QRect rect = clampRectToImage(QRect(m_startImagePoint, current).normalized());

        if (rect.width() >= kMinBoxSize && rect.height() >= kMinBoxSize) {
            pushUndoState();

            Annotation ann;
            ann.rect = rect;
            ann.label = m_currentLabel.isEmpty() ? "object" : m_currentLabel;

            annotations.append(ann);
            selectedIndex = annotations.size() - 1;
        }

        m_dragMode = DragMode::None;
        m_dragChanged = false;
        update();
        emit annotationsChanged();
        return;
    }

    if (m_dragMode == DragMode::Moving || m_dragMode == DragMode::Resizing) {
        if (!m_dragChanged && !m_undoStack.isEmpty()) {
            // 只是单击选择，没有真正移动/缩放；撤销栈中刚压入的状态可以移除。
            m_undoStack.removeLast();
        }

        m_dragMode = DragMode::None;
        m_resizeHandle = ResizeHandle::None;
        m_dragChanged = false;
        updateHoverCursor(event->pos());
        update();
        emit annotationsChanged();
    }
}

void ImageCanvas::wheelEvent(QWheelEvent *event)
{
    if (m_image.isNull()) {
        event->ignore();
        return;
    }

    if (event->angleDelta().y() > 0) {
        applyZoom(1.15, event->position());
    } else if (event->angleDelta().y() < 0) {
        applyZoom(1.0 / 1.15, event->position());
    }

    event->accept();
}

void ImageCanvas::leaveEvent(QEvent *event)
{
    Q_UNUSED(event);

    if (m_dragMode == DragMode::None) {
        unsetCursor();
    }
}
