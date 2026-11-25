import os

from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")

if not BOT_TOKEN:
    raise ValueError("Не найден BOT_TOKEN. Проверь файл .env")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEMP_DIR = os.path.join(BASE_DIR, "temp")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODELS_DIR = os.path.join(BASE_DIR, "models")

for directory in (TEMP_DIR, OUTPUT_DIR, MODELS_DIR):
    os.makedirs(directory, exist_ok=True)

# Сколько фотографий обрабатываем одновременно.
# Каждое фото занимает одно ядро CPU на 2-4 секунды.
MAX_CONCURRENT_JOBS = max(1, min(4, (os.cpu_count() or 2) // 2))

# Ограничение Telegram Bot API на скачивание файла.
MAX_INPUT_BYTES = 20 * 1024 * 1024

# Telegram сжимает фотографии, поэтому результат отдаём документом.
OUTPUT_JPEG_QUALITY = 95
