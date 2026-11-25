import asyncio
import logging
import os
import sys
import time

from aiogram import Bot, Dispatcher
from aiogram.types import ErrorEvent

from config import BOT_TOKEN, OUTPUT_DIR, TEMP_DIR
from handlers.photo import router as photo_router
from handlers.start import router as start_router
from processing.depth import warmup

logger = logging.getLogger(__name__)

STALE_FILE_AGE = 24 * 60 * 60


def clean_stale_files() -> None:
    """Убирает файлы, оставшиеся от прошлых запусков."""
    cutoff = time.time() - STALE_FILE_AGE

    for directory in (TEMP_DIR, OUTPUT_DIR):
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_file() and entry.stat().st_mtime < cutoff:
                        os.remove(entry.path)
        except OSError:
            logger.warning("Не удалось очистить %s", directory)


async def on_error(event: ErrorEvent) -> None:
    logger.exception("Необработанная ошибка", exc_info=event.exception)

    update = event.update
    message = update.message if update else None

    if message is not None:
        try:
            await message.answer(
                "❌ Что-то пошло не так. Попробуй ещё раз."
            )
        except Exception:  # noqa: BLE001 - сообщение может быть недоступно
            pass


async def warm_up_model() -> None:
    """Готовим модель глубины заранее, чтобы первая фотография не ждала."""
    try:
        ready = await asyncio.to_thread(warmup)
    except Exception:  # noqa: BLE001
        logger.exception("Не удалось подготовить модель глубины")
        return

    if ready:
        logger.info("Модель глубины готова")
    else:
        logger.warning(
            "Модель глубины недоступна: вспышка будет без учёта рельефа"
        )


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        stream=sys.stdout,
    )

    bot = Bot(token=BOT_TOKEN)
    dispatcher = Dispatcher()

    dispatcher.include_router(start_router)
    dispatcher.include_router(photo_router)
    dispatcher.errors.register(on_error)

    logger.info("G7X AI PHOTO запущен")

    clean_stale_files()

    warmup_task = asyncio.create_task(warm_up_model())

    try:
        await dispatcher.start_polling(
            bot,
            allowed_updates=dispatcher.resolve_used_update_types(),
        )
    finally:
        warmup_task.cancel()
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот остановлен")
