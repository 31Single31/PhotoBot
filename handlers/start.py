from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

router = Router(name="start")

GREETING = (
    "📸 G7X AI PHOTO\n\n"
    "Превращаю обычную фотографию в кадр со вспышкой Canon G7X: "
    "свет из центра кадра, падающая яркость в глубину, "
    "нарисованные тени и объём.\n\n"
    "Как пользоваться:\n"
    "1. Пришли фотографию\n"
    "2. Подожди пару секунд\n"
    "3. Забери готовый кадр\n\n"
    "Совет: если отправить фото файлом (без сжатия), "
    "качество результата будет выше."
)

HELP = (
    "Коротко о том, что происходит с фотографией:\n\n"
    "🔦 Вспышка светит из центра кадра, как встроенная в камеру.\n"
    "🌒 Дальние объекты получают меньше света — появляется глубина.\n"
    "🕯 Тени падают от объектов на фон, а в углублениях копится темнота.\n"
    "🎞 Цветокоррекция, зерно и виньетка доводят кадр до плёночного вида.\n\n"
    "Лица и силуэты не ищутся: работает любая фотография — "
    "портрет, улица, еда, животные."
)


@router.message(CommandStart())
async def start_handler(message: Message) -> None:
    await message.answer(GREETING)


@router.message(Command("help"))
async def help_handler(message: Message) -> None:
    await message.answer(HELP)


@router.message(F.text & ~F.text.startswith("/"))
async def text_handler(message: Message) -> None:
    await message.answer(
        "Пришли фотографию — и я наложу вспышку G7X. "
        "Можно отправить файлом, так качество выше."
    )
