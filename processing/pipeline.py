"""Сборка эффекта: глубина -> свет вспышки -> тон и цвет.

На вход одна фотография. Никаких детекторов: вспышка всегда считается
от центра кадра, а рельеф сцены берётся из карты глубины.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .depth import estimate_depth
from .grade import GradeSettings, develop
from .imaging import resize_to_long_side, unsharp
from .relight import FlashSettings, relight


@dataclass(frozen=True)
class ProcessingSettings:
    # Рабочее разрешение: компромисс между качеством и скоростью.
    work_side: int = 1500

    # Итоговое увеличение с запасом под сжатие Telegram.
    output_scale: float = 1.5
    max_output_side: int = 2600

    flash: FlashSettings = field(default_factory=FlashSettings)
    grade: GradeSettings = field(default_factory=GradeSettings)


DEFAULT_SETTINGS = ProcessingSettings()


def process_photo(
    image_bgr: np.ndarray,
    settings: ProcessingSettings | None = None,
) -> np.ndarray:
    """Применяет эффект вспышки Canon G7X. Возвращает BGR uint8."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Получено пустое изображение")

    config = settings or DEFAULT_SETTINGS

    work = resize_to_long_side(image_bgr, config.work_side)
    work = np.ascontiguousarray(work, dtype=np.uint8)

    depth = estimate_depth(work)

    linear = relight(work, depth, config.flash)
    developed = develop(linear, config.grade)

    height, width = developed.shape[:2]
    factor = min(
        config.output_scale,
        config.max_output_side / float(max(height, width)),
    )

    if factor <= 1.0:
        return developed

    resized = cv2.resize(
        developed,
        (int(round(width * factor)), int(round(height * factor))),
        interpolation=cv2.INTER_LANCZOS4,
    )

    return (
        unsharp(resized.astype(np.float32) / 255.0, 0.8, 0.10) * 255.0
    ).round().clip(0, 255).astype(np.uint8)
