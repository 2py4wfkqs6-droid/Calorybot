import os
import json
import logging
import sqlite3
import asyncio
from datetime import date
from io import BytesIO

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, CallbackQuery
from aiogram.utils.keyboard import InlineKeyboardBuilder
from dotenv import load_dotenv
from openai import OpenAI, APIStatusError, APITimeoutError, APIConnectionError
import base64
import re
from collections import deque
from aiohttp import web

load_dotenv()
logging.basicConfig(level=logging.INFO)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY", "")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID", "")
YANDEX_MODEL = os.getenv("YANDEX_MODEL", "qwen3.6-35b-a3b")
FREE_SCANS_PER_DAY = int(os.getenv("FREE_SCANS_PER_DAY", "10"))
DATABASE_PATH = os.getenv("DATABASE_PATH", "calorie_bot.sqlite3")
PORT = int(os.getenv("PORT", "8080"))
WEBHOOK_BASE_URL = (os.getenv("WEBHOOK_BASE_URL") or os.getenv("RENDER_EXTERNAL_URL") or "").rstrip("/")
WEBHOOK_PATH = "/telegram-webhook"
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")

if not BOT_TOKEN:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN in environment variables")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()



def connect_db():
    return sqlite3.connect(DATABASE_PATH)


def init_db():
    with connect_db() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            daily_goal INTEGER NOT NULL DEFAULT 2000
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS meals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            day TEXT NOT NULL,
            name TEXT NOT NULL,
            grams INTEGER NOT NULL,
            calories INTEGER NOT NULL,
            protein REAL NOT NULL,
            fat REAL NOT NULL,
            carbs REAL NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""")


def ensure_user(user_id):
    with connect_db() as conn:
        conn.execute("INSERT OR IGNORE INTO users(user_id) VALUES (?)", (user_id,))


def get_goal(user_id):
    ensure_user(user_id)
    with connect_db() as conn:
        return conn.execute("SELECT daily_goal FROM users WHERE user_id=?", (user_id,)).fetchone()[0]


def scans_today(user_id):
    with connect_db() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM meals WHERE user_id=? AND day=?",
            (user_id, date.today().isoformat())
        ).fetchone()[0]


def save_meal(user_id, meal):
    with connect_db() as conn:
        conn.execute(
            """INSERT INTO meals(user_id, day, name, grams, calories, protein, fat, carbs)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, date.today().isoformat(), meal["name"], meal["grams"],
             meal["calories"], meal["protein_g"], meal["fat_g"], meal["carbs_g"])
        )


def get_summary(user_id):
    with connect_db() as conn:
        return conn.execute(
            """SELECT COALESCE(SUM(calories),0), COALESCE(SUM(protein),0),
                      COALESCE(SUM(fat),0), COALESCE(SUM(carbs),0), COUNT(*)
               FROM meals WHERE user_id=? AND day=?""",
            (user_id, date.today().isoformat())
        ).fetchone()


def analyse_food(image_bytes):
    prompt = """
Ты анализируешь еду для дневника питания. Верни только корректный JSON без markdown:
{
 "name": "короткое название блюда",
 "estimated_items": ["видимые продукты"],
 "grams": 250,
 "calories": 450,
 "protein_g": 25,
 "fat_g": 15,
 "carbs_g": 50,
 "confidence": "низкая|средняя|высокая",
 "assumptions": "коротко об оценке порции и неопределённости"
}
Оцени примерные калории, белки, жиры и углеводы для видимой порции. Не утверждай,
что вес можно точно узнать по фото. Если вес или состав неясны, укажи это в assumptions.
Учитывай масло и соусы только если они видны или разумно предполагаются, и отмечай неопределённость.
Значения должны быть правдоподобными и согласованными. Не давай медицинских советов.
"""
    if not YANDEX_API_KEY or not YANDEX_FOLDER_ID:
        raise RuntimeError("Add YANDEX_API_KEY and YANDEX_FOLDER_ID in Render Environment")
    # Use the Yandex OpenAI-compatible Responses API for image understanding.
    # Disable SDK auto-retries to avoid multi-minute waits on 500 errors.
    client = OpenAI(
        api_key=YANDEX_API_KEY,
        base_url="https://ai.api.cloud.yandex.net/v1",
        project=YANDEX_FOLDER_ID,
        timeout=35.0,
        max_retries=0,
    )
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    response = client.responses.create(
        model=f"gpt://{YANDEX_FOLDER_ID}/{YANDEX_MODEL}",
        input=[{"role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}"},
        ]}],
        temperature=0.2,
    )
    raw = response.output_text or ""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
    if not raw.startswith("{"):
        raw = raw[raw.find("{"):raw.rfind("}")+1]
    result = json.loads(raw)
    required = ["name", "grams", "calories", "protein_g", "fat_g", "carbs_g", "confidence", "assumptions"]
    for key in required:
        if key not in result:
            raise ValueError(f"Missing field: {key}")
    result["grams"] = max(0, round(float(result["grams"])))
    result["calories"] = max(0, round(float(result["calories"])))
    for key in ("protein_g", "fat_g", "carbs_g"):
        result[key] = max(0, round(float(result[key]), 1))
    return result


def keyboard():
    kb = InlineKeyboardBuilder()
    kb.button(text="📊 Мой день", callback_data="today")
    kb.button(text="🎯 Цель калорий", callback_data="goal")
    kb.adjust(2)
    return kb.as_markup()


@dp.message(CommandStart())
async def start(message: Message):
    ensure_user(message.from_user.id)
    await message.answer(
        "Привет! Я тестовый AI-счётчик калорий 🍽️\n\n"
        "Отправь фото блюда - я оценю калории, белки, жиры и углеводы. "
        "В тестовой версии доступно до 10 фото в день.\n\n"
        "Команды: /today - дневник, /goal 2000 - цель, /help - инструкция.\n\n"
        "Важно: результат приблизительный. Фото не показывает точно вес, масло и скрытые ингредиенты.",
        reply_markup=keyboard()
    )


@dp.message(Command("help"))
async def help_cmd(message: Message):
    await message.answer(
        "Отправь фотографию еды. Я оценю блюдо и автоматически добавлю его в дневник.\n"
        "Для более точной оценки можно написать примерный вес порции в подписи к фото.\n"
        "Эта версия предназначена для тестирования и не заменяет рекомендации врача или диетолога."
    )


@dp.message(Command("goal"))
async def goal_cmd(message: Message):
    parts = (message.text or "").split()
    if len(parts) != 2 or not parts[1].isdigit() or not 500 <= int(parts[1]) <= 7000:
        await message.answer("Пример: /goal 2000 (от 500 до 7000 ккал)")
        return
    ensure_user(message.from_user.id)
    with connect_db() as conn:
        conn.execute("UPDATE users SET daily_goal=? WHERE user_id=?",
                     (int(parts[1]), message.from_user.id))
    await message.answer(f"Цель сохранена: {parts[1]} ккал в день.")


async def send_today(message: Message, user_id: int):
    calories, protein, fat, carbs, count = get_summary(user_id)
    goal = get_goal(user_id)
    await message.answer(
        f"📊 Дневник за {date.today().strftime('%d.%m.%Y')}\n\n"
        f"🔥 Калории: {calories} / {goal} ккал\n"
        f"До цели: {max(0, goal-calories)} ккал\n"
        f"🥩 Белки: {protein:.1f} г\n🥑 Жиры: {fat:.1f} г\n"
        f"🍞 Углеводы: {carbs:.1f} г\n\nБлюд добавлено: {count}",
        reply_markup=keyboard()
    )


@dp.message(Command("today"))
async def today_cmd(message: Message):
    await send_today(message, message.from_user.id)


@dp.callback_query(F.data == "today")
async def today_button(callback: CallbackQuery):
    await send_today(callback.message, callback.from_user.id)
    await callback.answer()


@dp.callback_query(F.data == "goal")
async def goal_button(callback: CallbackQuery):
    await callback.message.answer("Установи цель командой, например: /goal 2000")
    await callback.answer()


@dp.message(F.photo)
async def photo_handler(message: Message):
    user_id = message.from_user.id
    ensure_user(user_id)
    if scans_today(user_id) >= FREE_SCANS_PER_DAY:
        await message.answer(f"Тестовый лимит - {FREE_SCANS_PER_DAY} фото в сутки. Попробуй завтра.")
        return
    status = await message.answer("🔎 Анализирую фотографию...")
    try:
        photo = message.photo[-1]
        file = await bot.get_file(photo.file_id)
        buffer = BytesIO()
        await bot.download_file(file.file_path, destination=buffer)
        meal = await asyncio.to_thread(analyse_food, buffer.getvalue())
        save_meal(user_id, meal)
        summary = get_summary(user_id)
        items = ", ".join(meal.get("estimated_items", []))
        await status.edit_text(
            f"🍽️ {meal['name']}\n"
            f"{items + chr(10) if items else ''}"
            f"⚖️ Примерный вес: {meal['grams']} г\n"
            f"🔥 Калории: {meal['calories']} ккал\n"
            f"🥩 Белки: {meal['protein_g']} г\n"
            f"🥑 Жиры: {meal['fat_g']} г\n"
            f"🍞 Углеводы: {meal['carbs_g']} г\n\n"
            f"Уверенность: {meal['confidence']}\n"
            f"ℹ️ {meal['assumptions']}\n\n"
            f"Добавлено в дневник. Всего сегодня: {summary[0]} ккал.\n"
            "Оценка приблизительная - вес и скрытые ингредиенты могут отличаться.",
            reply_markup=keyboard()
        )
    except APIStatusError as exc:
        logging.error("Yandex API error: status=%s response=%s", exc.status_code, str(exc)[:1000])
        await status.edit_text(f"Сервис распознавания вернул ошибку {exc.status_code}. Попробуй позже.")
    except (APITimeoutError, APIConnectionError):
        logging.exception("Yandex API connection/timeout")
        await status.edit_text("Сервис распознавания не ответил вовремя. Попробуй позже.")
    except Exception:
        logging.exception("Photo analysis failed")
        await status.edit_text("Не удалось обработать фото. Подробности записаны в Render Logs.")


@dp.message()
async def other_message(message: Message):
    await message.answer("Отправь фото еды или выбери действие. Команда /help - инструкция.")


async def health(request):
    return web.Response(text="Calorie AI test bot is running")


# Telegram may resend the same update if webhook response is slow.
# Remember recent update IDs and immediately acknowledge incoming webhooks.
_seen_updates = set()
_seen_order = deque(maxlen=2000)
_background_tasks = set()


def _remember_update(update_id):
    if update_id in _seen_updates:
        return False
    if len(_seen_order) == _seen_order.maxlen:
        _seen_updates.discard(_seen_order.popleft())
    _seen_order.append(update_id)
    _seen_updates.add(update_id)
    return True


async def _process_update(update):
    try:
        from aiogram.types import Update
        await dp.feed_update(bot, Update.model_validate(update))
    except Exception:
        logging.exception("Webhook update processing failed")


async def telegram_webhook(request):
    if WEBHOOK_SECRET and request.headers.get("X-Telegram-Bot-Api-Secret-Token") != WEBHOOK_SECRET:
        raise web.HTTPUnauthorized()
    update = await request.json()
    update_id = update.get("update_id")
    if update_id is None or _remember_update(update_id):
        task = asyncio.create_task(_process_update(update))
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
    return web.Response(text="OK")


async def on_startup(app):
    init_db()
    if WEBHOOK_BASE_URL:
        await bot.set_webhook(
            WEBHOOK_BASE_URL + WEBHOOK_PATH,
            secret_token=WEBHOOK_SECRET or None,
            drop_pending_updates=True
        )
        logging.info("Webhook mode enabled")
    else:
        logging.info("No WEBHOOK_BASE_URL set; use local polling mode")


async def on_cleanup(app):
    if _background_tasks:
        await asyncio.gather(*_background_tasks, return_exceptions=True)
    if WEBHOOK_BASE_URL:
        await bot.delete_webhook()
    await bot.session.close()


def create_app():
    app = web.Application()
    app.router.add_get("/", health)
    app.router.add_post(WEBHOOK_PATH, telegram_webhook)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


async def main():
    init_db()
    if WEBHOOK_BASE_URL:
        app = create_app()
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", PORT)
        await site.start()
        logging.info("Webhook server listening on %s", PORT)
        await asyncio.Event().wait()
    else:
        logging.warning("No public URL detected; starting polling mode")
        await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
