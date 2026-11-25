from __future__ import annotations

import asyncio
import logging
import os
import uuid

import cv2

from aiogram import Bot, F, Router
from aiogram.types import FSInputFile, Message

from config import (
    MAX_CONCURRENT_JOBS,
    MAX_INPUT_BYTES,
    OUTPUT_DIR,
    OUTPUT_JPEG_QUALITY,
    TEMP_DIR,
)
from processing.depth import is_model_ready
from processing.pipeline import process_photo

router = Router(name="photo")
logger = logging.getLogger(__name__)

_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

STATUS_TEXT = "⚡ Считаю глубину кадра и ставлю вспышку…"

STATUS_FIRST_RUN = (
    "⚡ Первый запуск: скачиваю модель глубины (~95 МБ).\n"
    "Это нужно один раз."
)

CAPTION = (
    "📸 Готово\n\n"
    "⚡ Вспышка от центра кадра\n"
    "🌒 Тени и объём — по глубине снимка\n"
    "🎞 Цвет и зерно Canon G7X"
)

UNSUPPORTED_TEXT = (
    "❌ Не удалось открыть это как изображение.\n"
    "Подойдут JPEG, PNG или WebP."
)

FAILURE_TEXT = (
    "❌ Не получилось обработать фотографию.\n"
    "Попробуй другую или пришли файлом без сжатия."
)


class UnsupportedImage(ValueError):
    """Файл не удалось прочитать как изображение."""


def _render(input_path: str, output_path: str) -> None:
    """Тяжёлая часть: открыть, обработать, сохранить. Идёт в отдельном потоке."""
    image = cv2.imread(input_path, cv2.IMREAD_COLOR)

    if image is None or image.size == 0:
        raise UnsupportedImage("Файл не является изображением")

    height, width = image.shape[:2]

    if height < 32 or width < 32:
        raise UnsupportedImage("Слишком маленькое изображение")

    result = process_photo(image)

    if not cv2.imwrite(
        output_path,
        result,
        [cv2.IMWRITE_JPEG_QUALITY, OUTPUT_JPEG_QUALITY],
    ):
        raise RuntimeError("Не удалось сохранить результат")


async def _delete_quietly(message: Message) -> None:
    try:
        await message.delete()
    except Exception:  # noqa: BLE001 - сообщение могло уже исчезнуть
        pass


async def _handle_media(
    message: Message,
    bot: Bot,
    file_id: str,
    extension: str,
) -> None:
    unique_id = uuid.uuid4().hex
    input_path = os.path.join(TEMP_DIR, f"{unique_id}_input.{extension}")
    output_path = os.path.join(OUTPUT_DIR, f"{unique_id}_g7x.jpg")

    status = await message.answer(
        STATUS_TEXT if is_model_ready() else STATUS_FIRST_RUN
    )

    try:
        telegram_file = await bot.get_file(file_id)
        await bot.download_file(telegram_file.file_path, destination=input_path)

        async with _semaphore:
            await asyncio.to_thread(_render, input_path, output_path)

        await message.answer_document(
            FSInputFile(output_path, filename=f"g7x_{unique_id[:8]}.jpg"),
            caption=CAPTION,
        )
        await _delete_quietly(status)

    except UnsupportedImage:
        logger.info("Пропуск файла: не изображение")
        await status.edit_text(UNSUPPORTED_TEXT)

    except Exception:
        logger.exception("Ошибка обработки фотографии")
        await status.edit_text(FAILURE_TEXT)

    finally:
        for path in (input_path, output_path):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                logger.warning("Не удалось удалить %s", path)


@router.message(F.photo)
async def photo_handler(message: Message, bot: Bot) -> None:
    photo = message.photo[-1]

    if photo.file_size and photo.file_size > MAX_INPUT_BYTES:
        await message.answer("❌ Файл слишком большой.")
        return

    await _handle_media(message, bot, photo.file_id, "jpg")


@router.message(F.document.mime_type.startswith("image/"))
async def image_document_handler(message: Message, bot: Bot) -> None:
    document = message.document

    if document.file_size and document.file_size > MAX_INPUT_BYTES:
        await message.answer("❌ Файл слишком большой.")
        return

    extension = os.path.splitext(document.file_name or "")[1].lstrip(".").lower()

    await _handle_media(message, bot, document.file_id, extension or "jpg")
