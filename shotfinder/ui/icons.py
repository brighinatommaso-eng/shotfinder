"""Small painted icons: the app/tray icon and placeholder glyphs.

No bitmap assets are shipped; everything is drawn with QPainter so the app
stays a single Python package and the icons stay crisp at any DPI.
"""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QIcon,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)


def app_icon(size: int = 64) -> QIcon:
    """A blue rounded square with a lens: readable at 16 px in the tray."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    radius = size * 0.23
    background = QPainterPath()
    background.addRoundedRect(QRectF(1, 1, size - 2, size - 2), radius, radius)
    gradient = QLinearGradient(0, 0, size, size)
    gradient.setColorAt(0.0, QColor("#5BC8FF"))
    gradient.setColorAt(1.0, QColor("#0A66C2"))
    painter.fillPath(background, gradient)

    pen = QPen(QColor(255, 255, 255))
    pen.setWidthF(size * 0.085)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawEllipse(
        QRectF(size * 0.24, size * 0.22, size * 0.36, size * 0.36)
    )
    painter.drawLine(
        QPointF(size * 0.58, size * 0.56),
        QPointF(size * 0.77, size * 0.75),
    )
    painter.end()
    return QIcon(pixmap)


def glyph(kind: str, color: str, size: int = 48) -> QPixmap:
    """Line glyph for the empty state (``search``, ``image`` or ``video``)."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(color))
    pen.setWidthF(max(1.5, size * 0.055))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    box = QRectF(size * 0.12, size * 0.12, size * 0.76, size * 0.76)
    if kind == "search":
        painter.drawEllipse(QRectF(size * 0.16, size * 0.14, size * 0.52, size * 0.52))
        painter.drawLine(
            QPointF(size * 0.62, size * 0.62), QPointF(size * 0.84, size * 0.84)
        )
    elif kind == "video":
        painter.drawRoundedRect(box, size * 0.12, size * 0.12)
        painter.drawLine(
            QPointF(size * 0.42, size * 0.36), QPointF(size * 0.66, size * 0.5)
        )
        painter.drawLine(
            QPointF(size * 0.66, size * 0.5), QPointF(size * 0.42, size * 0.64)
        )
        painter.drawLine(
            QPointF(size * 0.42, size * 0.64), QPointF(size * 0.42, size * 0.36)
        )
    else:
        painter.drawRoundedRect(box, size * 0.12, size * 0.12)
        painter.drawEllipse(QRectF(size * 0.28, size * 0.28, size * 0.14, size * 0.14))
        painter.drawLine(
            QPointF(size * 0.2, size * 0.72), QPointF(size * 0.44, size * 0.48)
        )
        painter.drawLine(
            QPointF(size * 0.44, size * 0.48), QPointF(size * 0.62, size * 0.66)
        )
        painter.drawLine(
            QPointF(size * 0.62, size * 0.66), QPointF(size * 0.78, size * 0.5)
        )
    painter.end()
    return pixmap


def placeholder(size: tuple[int, int], label: str, dark: bool) -> QPixmap:
    """Neutral thumbnail placeholder showing the file extension."""
    width, height = size
    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    base = QColor(48, 48, 48) if dark else QColor(232, 232, 232)
    gradient = QLinearGradient(0, 0, 0, height)
    gradient.setColorAt(0.0, base.lighter(108))
    gradient.setColorAt(1.0, base.darker(108))
    path = QPainterPath()
    path.addRoundedRect(QRectF(0, 0, width, height), 8, 8)
    painter.fillPath(path, gradient)

    pen = QPen(QColor(255, 255, 255, 60) if dark else QColor(0, 0, 0, 40))
    pen.setWidthF(1.2)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawPath(path)

    font = painter.font()
    font.setPointSizeF(max(7.0, height * 0.12))
    font.setBold(True)
    font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.2)
    painter.setFont(font)
    painter.setPen(QColor(255, 255, 255, 130) if dark else QColor(0, 0, 0, 110))
    painter.drawText(
        QRectF(0, 0, width, height), Qt.AlignmentFlag.AlignCenter, label.upper()[:5]
    )
    painter.end()
    return pixmap
