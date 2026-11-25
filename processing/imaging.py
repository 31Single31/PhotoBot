"""Низкоуровневые операции над изображением.

Все функции работают с float32 в диапазоне 0..1, если не указано иное.
"""

from __future__ import annotations

import cv2
import numpy as np


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    """Перевод sRGB (0..1) в линейное пространство света."""
    image = np.clip(image, 0.0, 1.0)
    return np.where(
        image <= 0.04045,
        image / 12.92,
        ((image + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)


_SRGB8_TO_LINEAR = srgb_to_linear(
    np.arange(256, dtype=np.float32) / 255.0
).astype(np.float32)


def srgb_to_linear_u8(image: np.ndarray) -> np.ndarray:
    """То же, но по таблице: заметно быстрее для 8-битных снимков."""
    return _SRGB8_TO_LINEAR[image]


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    """Обратный перевод из линейного пространства в sRGB (0..1)."""
    image = np.clip(image, 0.0, 1.0)
    return np.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * np.power(image, 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def luminance(image: np.ndarray) -> np.ndarray:
    """Яркость по Rec.709 для массива HxWx3."""
    return (
        image[..., 0] * 0.2126
        + image[..., 1] * 0.7152
        + image[..., 2] * 0.0722
    ).astype(np.float32)


def gaussian(image: np.ndarray, sigma: float) -> np.ndarray:
    """Размытие по Гауссу, устойчивое к sigma=0."""
    if sigma <= 0:
        return image
    return cv2.GaussianBlur(image, (0, 0), sigma)


def box(image: np.ndarray, radius: int) -> np.ndarray:
    size = radius * 2 + 1
    return cv2.boxFilter(image, -1, (size, size), borderType=cv2.BORDER_REFLECT)


def guided_filter(
    guide: np.ndarray,
    source: np.ndarray,
    radius: int,
    epsilon: float,
) -> np.ndarray:
    """Edge-aware сглаживание source по границам guide.

    Нужно, чтобы низкое разрешение карты глубины «прилипало»
    к настоящим контурам объектов на фотографии.
    """
    guide = guide.astype(np.float32)
    source = source.astype(np.float32)

    mean_guide = box(guide, radius)
    mean_source = box(source, radius)
    corr_guide = box(guide * guide, radius)
    corr_cross = box(guide * source, radius)

    var_guide = corr_guide - mean_guide * mean_guide
    cov_cross = corr_cross - mean_guide * mean_source

    a = cov_cross / (var_guide + epsilon)
    b = mean_source - a * mean_guide

    return box(a, radius) * guide + box(b, radius)


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    if edge1 <= edge0:
        return (x >= edge1).astype(np.float32)
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def resize_to_long_side(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)

    if longest <= max_side:
        return image

    scale = max_side / float(longest)
    return cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def unsharp(
    image: np.ndarray,
    sigma: float,
    amount: float,
    threshold: float = 0.0,
) -> np.ndarray:
    """Контурное усиление с порогом, чтобы не разгонять шум в тенях."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    detail = image - blurred

    if threshold > 0:
        detail *= smoothstep(0.0, threshold, np.abs(detail))

    return np.clip(image + detail * amount, 0.0, 1.0).astype(np.float32)
