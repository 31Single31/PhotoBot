"""Имитация прямой вспышки по карте глубины.

Модель простая и физичная: вспышка стоит рядом с объективом и светит
на сцену. Освещённость падает как 1/d^2, поэтому ближний план
вырывается из темноты, а дальний уходит в тень. По карте глубины
считаются нормали поверхности (объём, светотень), отброшенные тени
и затемнение углублений.

Никаких лиц, силуэтов и масок: свет всегда идёт из центра кадра.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .imaging import gaussian, luminance, smoothstep, srgb_to_linear_u8


@dataclass(frozen=True)
class FlashSettings:
    # Геометрия сцены в метрах (условных, важна только разница).
    near_distance: float = 0.55
    far_distance: float = 6.0

    # Сила спада света с расстоянием. 1.0 — закон обратных квадратов.
    falloff: float = 1.25

    # Доля рассеянного света: чем меньше, тем чернее фон.
    ambient: float = 0.10

    # Спад яркости к краям кадра: до углов свет вспышки не достаёт.
    beam: float = 0.65

    # Положение вспышки в кадре (доли от половины ширины/высоты).
    light_x: float = 0.0
    light_y: float = -0.22
    light_depth: float = 0.30

    # Отброшенные тени.
    shadow_strength: float = 0.80
    shadow_bias: float = 0.05
    shadow_softness: float = 2.4

    # Светотень по нормалям (объём).
    relief: float = 0.35
    relief_smoothing: float = 150.0
    fill: float = 0.65

    # Блики.
    specular: float = 0.10
    specular_power: float = 26.0

    # Затемнение в углублениях.
    occlusion: float = 0.25


def _unit(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.maximum(
        np.linalg.norm(vectors, axis=2, keepdims=True),
        1e-6,
    )


def _distance_map(depth01: np.ndarray, settings: FlashSettings) -> np.ndarray:
    """Переводит 0..1 (1 — близко) в условные метры до камеры."""
    near = settings.near_distance
    far = settings.far_distance
    return (near + (far - near) * (1.0 - depth01)).astype(np.float32)


def _light_position(
    settings: FlashSettings,
    width: int,
    height: int,
    focal: float,
) -> np.ndarray:
    """Положение вспышки в метрах: X вправо, Y вниз, Z вглубь кадра."""
    x_norm = (settings.light_x * width) / focal
    y_norm = (settings.light_y * height) / focal

    return np.array(
        [
            x_norm * settings.light_depth,
            y_norm * settings.light_depth,
            settings.light_depth,
        ],
        dtype=np.float32,
    )


def _gradient(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grad_x = cv2.Sobel(field, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    grad_y = cv2.Sobel(field, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return grad_x, grad_y


def _fill_near_surfaces(distance: np.ndarray, radius: int) -> np.ndarray:
    """Расширяет ближний план на ширину размытой границы глубины.

    Карта глубины «размазывает» контур объекта на десяток пикселей.
    Если считать нормали по ней, вокруг объекта появляется валик из
    наклонённых нормалей. Расширив ближнюю поверхность на эту ширину,
    мы получаем ровный фон сразу за контуром.
    """
    if radius <= 1:
        return distance

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    return cv2.erode(distance, kernel)


def _shading_maps(
    distance: np.ndarray,
    focal: float,
    settings: FlashSettings,
) -> tuple[np.ndarray, np.ndarray]:
    """Нормали поверхности и маска разрывов глубины.

    На контуре объекта производная глубины огромна: нормаль «заваливается»
    и рисует тёмную обводку. Такие места помечаются, чтобы светотень
    и затемнение углублений там не применялись.
    """
    height, width = distance.shape
    smoothing = max(3.0, max(height, width) / settings.relief_smoothing)
    filled = _fill_near_surfaces(distance, int(max(4, max(height, width) / 38)))
    smooth = gaussian(filled, smoothing)

    # Разрыв ищем по неразмытой карте: на сглаженной граница «расползается».
    raw = gaussian(distance, 1.5)
    raw_x, raw_y = _gradient(raw)
    slope = np.sqrt(raw_x * raw_x + raw_y * raw_y) * focal
    slope /= np.maximum(raw, 1e-3)

    discontinuity = smoothstep(0.35, 1.20, slope)

    # Расширяем маску на ширину самой границы.
    radius = max(4, int(round(max(height, width) / 110)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    discontinuity = cv2.dilate(discontinuity, kernel)
    discontinuity = gaussian(discontinuity, radius * 0.4)

    grad_x, grad_y = _gradient(smooth)
    grad_x *= focal * settings.relief
    grad_y *= focal * settings.relief

    normal = np.dstack([grad_x, grad_y, -np.ones_like(grad_x)])
    return _unit(normal).astype(np.float32), discontinuity


def _cast_shadow(
    distance: np.ndarray,
    light: np.ndarray,
    focal: float,
    center_x: float,
    center_y: float,
    settings: FlashSettings,
) -> np.ndarray:
    """Отброшенные тени: экранный трассировщик от точки вспышки.

    Для каждой точки кадра проверяем, не перекрыт ли путь луча от
    вспышки до неё чем-то более близким к камере. Возвращает 1 там,
    где свет доходит, и 0 в тени.

    Считаем в половинном разрешении: тени всё равно мягкие, а скорость
    вырастает вчетверо.
    """
    height, width = distance.shape

    factor = 2 if min(height, width) >= 480 else 1

    if factor > 1:
        distance = cv2.resize(
            distance,
            (width // factor, height // factor),
            interpolation=cv2.INTER_AREA,
        )
        height, width = distance.shape
        focal /= factor
        center_x /= factor
        center_y /= factor

    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)

    point_x = ((xs - center_x) / focal) * distance
    point_y = ((ys - center_y) / focal) * distance
    point_z = distance

    light_x, light_y, light_z = light

    occlusion = np.zeros((height, width), dtype=np.float32)

    steps = 12
    for index in range(1, steps + 1):
        t = index / steps

        ray_z = light_z + t * (point_z - light_z)
        ray_x = light_x + t * (point_x - light_x)
        ray_y = light_y + t * (point_y - light_y)

        sampled = cv2.remap(
            distance,
            (ray_x / ray_z) * focal + center_x,
            (ray_y / ray_z) * focal + center_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # Насколько поверхность на пути луча ближе к камере, чем сама точка.
        penetration = (ray_z - sampled) / np.maximum(ray_z, 1e-3)
        occlusion = np.maximum(
            occlusion,
            smoothstep(
                settings.shadow_bias * 0.4,
                settings.shadow_bias * 2.2,
                penetration,
            ),
        )

    shadow = 1.0 - settings.shadow_strength * occlusion
    return np.clip(shadow, 0.0, 1.0).astype(np.float32)


def _occlusion(
    depth01: np.ndarray,
    settings: FlashSettings,
) -> np.ndarray:
    """Затемнение углублений: складки, стыки, пространство под объектами.

    Локальный уровень глубины берётся медианой, а не средним: иначе
    близкий объект «затекает» на соседний фон и вокруг него появляется
    тёмное кольцо.
    """
    height, width = depth01.shape

    # Медиана вместо среднего: одинокая близкая поверхность не «затекает»
    # на соседний фон и не рисует вокруг объекта тёмное кольцо.
    kernel = int(max(5, min(41, round(max(height, width) / 60)))) | 1
    depth_u8 = (depth01 * 255.0).astype(np.uint8)
    surrounding = cv2.medianBlur(depth_u8, kernel).astype(np.float32) * (1.0 / 255.0)

    cavity = np.clip(surrounding - depth01, 0.0, None)
    cavity = gaussian(cavity, max(1.0, kernel * 0.35))
    cavity = cavity / (np.percentile(cavity, 98.0) + 1e-6)

    return (1.0 - settings.occlusion * smoothstep(0.0, 1.0, cavity)).astype(
        np.float32
    )


def relight(
    image: np.ndarray,
    depth01: np.ndarray,
    settings: FlashSettings,
) -> np.ndarray:
    """Освещает кадр вспышкой.

    image — BGR uint8, depth01 — 1 там, где объект близко.
    Возвращает линейное изображение float32.
    """
    height, width = image.shape[:2]

    focal = 0.87 * width
    center_x = width / 2.0
    center_y = height / 2.0

    depth01 = np.clip(depth01, 0.0, 1.0)
    distance = _distance_map(depth01, settings)
    light = _light_position(settings, width, height, focal)

    # ---- отброшенные тени ---------------------------------------------
    shadow = _cast_shadow(
        distance,
        light,
        focal,
        center_x,
        center_y,
        settings,
    )

    if shadow.shape != (height, width):
        shadow = cv2.resize(
            shadow,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

    shadow = gaussian(shadow, settings.shadow_softness)

    # ---- светотень по нормалям ----------------------------------------
    normals, discontinuity = _shading_maps(distance, focal, settings)

    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    x_norm = (xs - center_x) / focal
    y_norm = (ys - center_y) / focal

    surface_x = x_norm * distance
    surface_y = y_norm * distance

    to_light = _unit(
        np.dstack(
            [
                light[0] - surface_x,
                light[1] - surface_y,
                light[2] - distance,
            ]
        )
    )

    lambert = np.clip(np.sum(normals * to_light, axis=2), 0.0, 1.0)
    shade = settings.fill + (1.0 - settings.fill) * lambert

    # На разрывах глубины светотень не рисуем.
    shade += (1.0 - shade) * discontinuity

    # ---- спад света ----------------------------------------------------
    reference = np.percentile(distance, 20.0)
    falloff = np.power(
        np.clip(reference / np.maximum(distance, 1e-3), 0.0, 8.0),
        settings.falloff,
    )

    radius = np.sqrt(
        ((xs - center_x) / (width * 0.5)) ** 2
        + ((ys - center_y) / (height * 0.5)) ** 2
    )
    beam = np.exp(-settings.beam * radius * radius)

    occlusion = _occlusion(depth01, settings)

    illumination = falloff * beam * shadow * shade * occlusion
    illumination = gaussian(illumination, max(1.0, width / 700.0))

    # Свет приводим к единице на самом освещённом участке кадра:
    # так вспышка выглядит одинаково на тёмных и светлых снимках.
    illumination /= np.percentile(illumination, 97.0) + 1e-6
    illumination = np.clip(illumination, 0.0, 1.5)

    # ---- итоговый свет -------------------------------------------------
    gain = settings.ambient + (1.0 - settings.ambient) * illumination

    # Блики: маленькие, только на ярких гладких участках.
    view = _unit(np.dstack([-surface_x, -surface_y, -distance]))
    half = _unit(to_light + view)

    specular = np.power(
        np.clip(np.sum(normals * half, axis=2), 0.0, 1.0),
        settings.specular_power,
    )

    linear = srgb_to_linear_u8(image)

    brightness = smoothstep(0.05, 0.45, luminance(linear))
    specular *= brightness * settings.specular * shadow

    linear = linear * gain[..., np.newaxis]
    linear += specular[..., np.newaxis] * 0.5

    return linear.astype(np.float32)
