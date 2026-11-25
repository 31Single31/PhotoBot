"""Тон-маппинг и цветокоррекция под характер Canon G7X.

Здесь всё, что делает кадр «вспышечным»: киношный спад светов,
тёплые света и холодные тени, свечение вокруг пересветов, виньетка
объектива и зерно высокого ISO.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .imaging import gaussian, linear_to_srgb, luminance, smoothstep, unsharp


@dataclass(frozen=True)
class GradeSettings:
    # Экспозиция: уровень, до которого дотягиваются света кадра.
    exposure_target: float = 0.78

    # Контраст и цвет.
    contrast: float = 0.32
    saturation: float = 0.97
    warm_highlights: float = 0.030
    cool_shadows: float = 0.050
    black_point: float = 0.020

    # Свечение вокруг пересветов.
    bloom: float = 0.24
    bloom_threshold: float = 1.10
    bloom_radius: float = 0.018

    # Виньетка объектива.
    vignette: float = 0.32

    # Зерно.
    grain: float = 0.018

    # Резкость на выходе. Мягкая: вспышка и так даёт плотные контуры,
    # а сильное контурное усиление превращает кожу и волосы в «крахмал».
    sharpen: float = 0.30
    sharpen_radius: float = 1.0


def _aces(x: np.ndarray) -> np.ndarray:
    """Кривая ACES (аппроксимация Narkowicz): мягкий спад светов."""
    return np.clip(
        (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14),
        0.0,
        1.0,
    )


def auto_exposure(linear: np.ndarray, target: float) -> float:
    """Экспозиция по светам, как у камеры со вспышкой.

    Камера выставляет экспозицию так, чтобы самые светлые участки
    остались чуть ниже пересвета. Тогда ближний план получается
    плотным, а тени уходят в глубину.
    """
    reference = float(np.percentile(luminance(linear), 98.0))

    if reference <= 1e-5:
        return 1.0

    return float(np.clip(target / reference, 0.5, 2.5))


def develop(linear: np.ndarray, settings: GradeSettings) -> np.ndarray:
    """Линейный кадр -> готовое sRGB-изображение (BGR uint8)."""
    height, width = linear.shape[:2]

    linear = linear * auto_exposure(linear, settings.exposure_target)

    # Свечение вокруг пересветов: сенсор и объектив «размазывают» свет.
    highlights = np.clip(
        luminance(linear) - settings.bloom_threshold,
        0.0,
        None,
    )[:, :, np.newaxis] * linear

    radius = max(1.0, min(height, width) * settings.bloom_radius)
    linear = linear + gaussian(highlights, radius) * settings.bloom

    # Виньетка: свет вспышки не достаёт до углов кадра.
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    radius_map = np.sqrt(
        ((xs - width * 0.5) / (width * 0.5)) ** 2
        + ((ys - height * 0.5) / (height * 0.5)) ** 2
    )
    vignette = 1.0 - settings.vignette * smoothstep(0.35, 1.45, radius_map)
    linear = linear * vignette[:, :, np.newaxis]

    image = linear_to_srgb(_aces(np.clip(linear, 0.0, 8.0)))

    # ---- цветокоррекция ------------------------------------------------
    image = np.maximum(image - settings.black_point, 0.0) / (
        1.0 - settings.black_point
    )

    light = luminance(image)[:, :, np.newaxis]

    # Мягкая S-кривая: глубже тени, плотнее света.
    curved = light * light * (3.0 - 2.0 * light)
    image = image + (curved - light) * settings.contrast
    light = luminance(np.clip(image, 0.0, 1.0))[:, :, np.newaxis]

    # Раздельное тонирование: тёплые света, холодные тени.
    shadow_weight = (1.0 - np.clip(light * 2.2, 0.0, 1.0)) ** 2
    highlight_weight = np.clip((light - 0.35) / 0.65, 0.0, 1.0) ** 1.5

    image = image + shadow_weight * settings.cool_shadows * np.array(
        [-0.10, 0.02, 0.12], dtype=np.float32
    )
    image = image + highlight_weight * settings.warm_highlights * np.array(
        [0.25, 0.55, -0.05], dtype=np.float32
    )

    # Насыщенность.
    light = luminance(np.clip(image, 0.0, 1.0))[:, :, np.newaxis]
    image = light + (image - light) * settings.saturation
    image = np.clip(image, 0.0, 1.0)

    # ---- резкость ------------------------------------------------------
    image = unsharp(
        image,
        sigma=settings.sharpen_radius,
        amount=settings.sharpen,
        threshold=0.02,
    )

    # ---- зерно ---------------------------------------------------------
    if settings.grain > 0:
        noise = np.random.normal(0.0, 1.0, (height, width)).astype(np.float32)
        noise = gaussian(noise, 0.7)
        noise *= (1.0 - light[:, :, 0]) ** 1.4 + 0.12
        image = image + noise[:, :, np.newaxis] * settings.grain

        chroma = np.random.normal(
            0.0, 1.0, (height, width, 3)
        ).astype(np.float32)
        chroma = gaussian(chroma, 1.6)
        image = image + chroma * settings.grain * 0.25

    return np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)
