"""Локальная проверка эффекта без запуска Telegram-бота.

Пример:
    python tools/preview.py photo.jpg
    python tools/preview.py photo.jpg --compare --depth
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processing.depth import estimate_depth  # noqa: E402
from processing.pipeline import process_photo  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="путь к исходной фотографии")
    parser.add_argument(
        "-o",
        "--output",
        help="куда сохранить результат (по умолчанию рядом с исходником)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="сохранить картинку «до / после»",
    )
    parser.add_argument(
        "--depth",
        action="store_true",
        help="сохранить карту глубины",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)

    if image is None:
        print(f"Не удалось открыть файл: {args.image}")
        return 1

    stem = os.path.splitext(args.image)[0]
    output = args.output or f"{stem}_g7x.jpg"

    result = process_photo(image)
    cv2.imwrite(output, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Готово: {output}  {result.shape[1]}x{result.shape[0]}")

    if args.compare:
        height = result.shape[0]
        width = int(image.shape[1] * height / image.shape[0])
        original = cv2.resize(image, (width, height))
        canvas = np.hstack([original, result])
        compare_path = f"{stem}_compare.jpg"
        cv2.imwrite(compare_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"Сравнение: {compare_path}")

    if args.depth:
        depth = estimate_depth(image)
        depth_path = f"{stem}_depth.png"
        cv2.imwrite(depth_path, (depth * 255).astype(np.uint8))
        print(f"Глубина: {depth_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
