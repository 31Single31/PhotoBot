"""Оценка рельефа сцены (карта глубины).

Модель Depth Anything V2 Small в формате ONNX. Она нужна не для поиска
лица или силуэта, а для того, чтобы вспышка «понимала» рельеф кадра:
что стоит близко, что далеко, где поверхность развёрнута к камере,
а где уходит в глубину.

Если модель недоступна (нет onnxruntime, нет интернета при первом
запуске), возвращается ровная карта: эффект вспышки всё равно
применяется, просто без учёта геометрии сцены.
"""

from __future__ import annotations

import logging
import os
import shutil
import urllib.request

import cv2
import numpy as np

from .imaging import guided_filter

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILENAME = "depth_anything_v2_small.onnx"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

MODEL_URL = (
    "https://huggingface.co/onnx-community/depth-anything-v2-small/"
    "resolve/main/onnx/model.onnx"
)

# Сеть работает с размером, кратным 14.
INPUT_SIZE = 714
SIZE_MULTIPLE = 14

# Средние значения ImageNet: так модель обучалась.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_session = None
_load_failed = False


def is_model_ready() -> bool:
    """Модель уже лежит на диске и её не нужно скачивать."""
    return os.path.exists(MODEL_PATH)


def download_model() -> str:
    """Скачивает модель один раз. Возвращает путь к файлу."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        return MODEL_PATH

    logger.info("Скачивание модели глубины (%.0f МБ)...", 95)
    partial = MODEL_PATH + ".part"

    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
            with open(partial, "wb") as target:
                shutil.copyfileobj(response, target, length=1024 * 1024)

        os.replace(partial, MODEL_PATH)
        logger.info("Модель глубины скачана: %s", MODEL_PATH)

    except Exception:
        if os.path.exists(partial):
            try:
                os.remove(partial)
            except OSError:
                pass
        raise

    return MODEL_PATH


def _load_session():
    global _session, _load_failed

    if _session is not None:
        return _session

    if _load_failed:
        return None

    try:
        import onnxruntime as ort

        download_model()

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        options.log_severity_level = 3

        _session = ort.InferenceSession(
            MODEL_PATH,
            options,
            providers=["CPUExecutionProvider"],
        )
        logger.info("Модель глубины загружена")

    except Exception as error:  # noqa: BLE001 - нужен любой сбой
        _load_failed = True
        logger.warning(
            "Карта глубины недоступна (%s). "
            "Вспышка будет применена без учёта рельефа.",
            error,
        )
        return None

    return _session


def _predict(image: np.ndarray) -> np.ndarray:
    session = _load_session()

    if session is None:
        raise RuntimeError("Модель глубины недоступна")

    height, width = image.shape[:2]

    scale = INPUT_SIZE / float(max(height, width))
    input_height = max(
        SIZE_MULTIPLE,
        int(round(height * scale / SIZE_MULTIPLE)) * SIZE_MULTIPLE,
    )
    input_width = max(
        SIZE_MULTIPLE,
        int(round(width * scale / SIZE_MULTIPLE)) * SIZE_MULTIPLE,
    )

    resized = cv2.resize(
        image,
        (input_width, input_height),
        interpolation=cv2.INTER_AREA,
    )

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - MEAN) / STD
    blob = normalized.transpose(2, 0, 1)[np.newaxis, ...]

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: blob})[0]

    depth = np.asarray(output, dtype=np.float32).reshape(
        output.shape[-2], output.shape[-1]
    )

    return cv2.resize(depth, (width, height), interpolation=cv2.INTER_CUBIC)


def estimate_depth(image: np.ndarray, refine: bool = True) -> np.ndarray:
    """Карта обратной глубины 0..1: 1 — близко к камере, 0 — далеко.

    Дополнительно привязывается к контурам самой фотографии, чтобы
    границы объектов не «плыли».
    """
    if image is None or image.size == 0:
        raise ValueError("Получено пустое изображение")

    height, width = image.shape[:2]

    # Совсем мелкие картинки незачем гонять через сеть.
    if max(height, width) < 64:
        return np.zeros((height, width), dtype=np.float32)

    try:
        raw = _predict(image)
    except Exception as error:  # noqa: BLE001
        logger.debug("Оценка глубины не удалась: %s", error)
        return np.zeros((height, width), dtype=np.float32)

    low, high = np.percentile(raw, 1.0), np.percentile(raw, 99.0)

    if high - low < 1e-6:
        return np.zeros((height, width), dtype=np.float32)

    normalized = np.clip((raw - low) / (high - low), 0.0, 1.0)

    if refine:
        guide = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        radius = max(4, int(round(max(height, width) / 90)))
        normalized = guided_filter(
            guide,
            normalized,
            radius=radius,
            epsilon=0.002,
        )

    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def warmup() -> bool:
    """Загружает модель заранее, чтобы первая фотография не ждала."""
    return _load_session() is not None
