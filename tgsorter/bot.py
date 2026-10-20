from __future__ import annotations

import asyncio
import json
import logging
import shutil
import time
from dataclasses import asdict
from datetime import datetime

import httpx
from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.telegram import TelegramAPIServer
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command, CommandObject
from aiogram.types import (CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton, Message,
                           ReactionTypeEmoji, ReplyKeyboardMarkup, ReplyParameters)

from .classifier import UNSORTED, Classification
from .config import Settings
from .db import DB
from .gemini import AllKeysInvalid, BadRequest, ContentBlocked, GeminiClient, QuotaExhausted
from .keypool import KeyPool
from .media import HAVE_FFMPEG
from .pipeline import Deps, analyze
from .telegram_io import extract

log = logging.getLogger("tgsorter")

HELP = (
    "I sort what you send me into your own categories, using Gemini.\n\n"
    "Everything is done with buttons - tap ☰ Menu (or /start) at any time.\n\n"
    "1) 📂 Categories → ➕ Add category: give it a name and a description (your own definition).\n"
    "2) Pick where sorted items go (optional): make a group, enable Topics, add me as admin, then open a\n"
    "   category → 📌 Bind to a topic, and send any message inside the topic you want.\n"
    "3) Send or forward me anything - text, photos, voice, audio, videos, PDFs. Albums are sorted together.\n"
    "   Long audio is sampled across the whole recording and I tell you which time ranges matter.\n"
    "   Tap '✏️ Change category' under my answer if I got it wrong."
)
MENU_BTN = "☰ Menu"


class State:
    def __init__(self):
        self.albums: dict[str, list[Message]] = {}
        self.outage_notified = False
        self.pending: dict[int, tuple] = {}     # user_id -> ("add",) | ("add_desc", name) | ("edit", cat) | ("bind", cat)


# ───────────────────────── keyboards & delivery ─────────────────────────
def change_kb(item_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text="✏️ Change category", callback_data=f"mv:{item_id}")]])


def cats_kb(item_id: int, cats: list[dict]) -> InlineKeyboardMarkup:
    btns = [InlineKeyboardButton(text=c["name"], callback_data=f"sv:{item_id}:{i}") for i, c in enumerate(cats)]
    btns.append(InlineKeyboardButton(text=UNSORTED, callback_data=f"sv:{item_id}:u"))
    return InlineKeyboardMarkup(inline_keyboard=[btns[i:i + 2] for i in range(0, len(btns), 2)])


def _kb(rows: list[list[tuple[str, str]]]) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text=t, callback_data=d) for t, d in r] for r in rows])


def main_menu() -> InlineKeyboardMarkup:
    return _kb([[("📂 Categories", "m:cats"), ("➕ Add category", "m:add")],
                [("🕘 Recent", "m:recent"), ("📊 Status", "m:status")],
                [("🔑 Keys", "m:keys"), ("❓ Help", "m:help")]])


def reply_menu() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text=MENU_BTN)]], resize_keyboard=True, is_persistent=True)


BACK = [("⬅️ Menu", "m:home")]
CANCEL = _kb([[("✖️ Cancel", "m:cancel")]])


async def copy_to(bot: Bot, item: dict, cat: dict | None) -> dict | None:
    if not cat or not cat.get("dest_chat_id"):
        return None
    ids = json.loads(item["message_ids"])
    res = await bot.copy_messages(chat_id=cat["dest_chat_id"], from_chat_id=item["chat_id"], message_ids=ids,
                                  message_thread_id=cat.get("dest_thread_id"))
    return {"chat_id": cat["dest_chat_id"], "ids": [r.message_id for r in res]}


async def drop_copies(bot: Bot, copied_json: str | None) -> None:
    if not copied_json:
        return
    c = json.loads(copied_json)
    try:
        await bot.delete_messages(chat_id=c["chat_id"], message_ids=c["ids"])
    except TelegramAPIError as e:
        log.info("could not delete old copies: %s", e)


async def deliver(item: dict, cls: Classification, deps: Deps) -> None:
    bot, s = deps.bot, deps.settings
    unsure = cls.category == UNSORTED or cls.confidence < s.confidence_threshold
    cat = None if unsure else await deps.db.category(item["user_id"], cls.category)
    copied, warn = None, ""
    try:
        copied = await copy_to(bot, item, cat)
    except TelegramAPIError as e:
        warn = f"\n⚠️ Couldn't copy to the destination: {e.message if hasattr(e, 'message') else e}"
    dest = " → filed" if copied else (" (tag only - no destination bound)" if cat else "")
    head = f"❓ Not sure: best guess “{cls.category}” ({cls.confidence:.0%})" if unsure else \
           f"📂 {cls.category} ({cls.confidence:.0%}){dest}"
    ts = ("\n\n🕒 Where in the audio:\n" + "\n".join(f"• {x}" for x in cls.parts)) if cls.parts else ""
    text = f"{head}\n{cls.title}\n{cls.summary}{ts}{warn}"
    kb = cats_kb(item["id"], await deps.db.categories(item["user_id"])) if unsure else change_kb(item["id"])
    first = json.loads(item["message_ids"])[0]
    reply = await bot.send_message(item["chat_id"], text[:4000], reply_markup=kb,
                                   reply_parameters=ReplyParameters(message_id=first, allow_sending_without_reply=True))
    await deps.db.complete(item["id"], None if unsure else cls.category, cls.title, cls.summary, cls.confidence,
                           copied, reply.message_id)


async def fail_notice(item: dict, deps: Deps, why: str) -> None:
    first = json.loads(item["message_ids"])[0]
    cats = await deps.db.categories(item["user_id"])
    await deps.bot.send_message(
        item["chat_id"], f"⚠️ I couldn't analyse this ({why}). Pick a category yourself:",
        reply_markup=cats_kb(item["id"], cats),
        reply_parameters=ReplyParameters(message_id=first, allow_sending_without_reply=True))


# ───────────────────────── worker ─────────────────────────
async def process_one(deps: Deps, state: State) -> bool:
    """Claim and process a single queued item. Returns False if the queue was empty."""
    db, s = deps.db, deps.settings
    item = await db.claim_next()
    if not item:
        return False
    try:
        cls = await analyze(item, deps)
        state.outage_notified = False
        await deliver(item, cls, deps)
    except QuotaExhausted as e:
        await db.defer(item["id"], e.retry_after + 1)
        if not state.outage_notified:
            state.outage_notified = True
            when = datetime.fromtimestamp(time.time() + e.retry_after).strftime("%H:%M")
            await _safe_send(deps.bot, item["chat_id"],
                             f"⏳ All Gemini keys are rate-limited right now. Your items stay queued and "
                             f"are retried automatically (next attempt ≈ {when}). Adding another key "
                             f"from a different Google project speeds this up.")
    except AllKeysInvalid:
        await db.retry_or_fail(item["id"], "all keys invalid", 1)
        await _safe_send(deps.bot, item["chat_id"], "🔑 Every configured Gemini key was rejected as invalid. "
                                                    "Fix GEMINI_API_KEYS and restart me.")
    except ContentBlocked as e:
        await db.retry_or_fail(item["id"], str(e), 1)
        await fail_notice(item, deps, f"Gemini declined to analyse it: {e}")
    except Exception as e:                      # BadRequest and anything unexpected
        log.exception("item %s failed", item["id"])
        if not await db.retry_or_fail(item["id"], f"{type(e).__name__}: {e}", s.max_job_attempts):
            await fail_notice(item, deps, type(e).__name__)
    return True


async def worker(deps: Deps, state: State, stop: asyncio.Event) -> None:
    while not stop.is_set():
        if not await process_one(deps, state):
            await asyncio.sleep(1)


async def _safe_send(bot: Bot, chat_id: int, text: str) -> None:
    try:
        await bot.send_message(chat_id, text)
    except TelegramAPIError:
        pass


# ───────────────────────── handlers ─────────────────────────
def build_router(deps: Deps, state: State) -> tuple[Router, Router]:
    s, db, bot = deps.settings, deps.db, deps.bot
    public, r = Router(), Router()
    def allowed(ev) -> bool:
        return s.is_allowed(ev.from_user.id if ev.from_user else None)
    r.message.filter(allowed)
    r.callback_query.filter(allowed)

    @public.message(Command("start"))
    async def start(m: Message):
        if not s.is_allowed(m.from_user.id):
            await m.answer(f"This is a private bot. Your Telegram ID is {m.from_user.id} - ask the owner to "
                           f"add it to ALLOWED_USER_IDS.")
        else:
            await m.answer(HELP, reply_markup=reply_menu())
            await m.answer("Menu:", reply_markup=main_menu())

    @r.message(Command("help"))
    async def help_(m: Message):
        await m.answer(HELP, reply_markup=main_menu())

    @r.message(Command("menu"))
    @r.message(F.text == MENU_BTN)
    async def menu(m: Message):
        state.pending.pop(m.from_user.id, None)
        await m.answer("Menu:", reply_markup=main_menu())

    @r.message(Command("add"))
    async def add(m: Message, command: CommandObject):
        name, _, desc = (command.args or "").partition("|")
        name, desc = name.strip(), desc.strip()
        if not name or len(name) > 40 or name.lower() == UNSORTED.lower():
            return await m.answer("Usage: /add Name | what belongs here   (name ≤ 40 chars, not “Unsorted”)")
        await db.upsert_category(m.from_user.id, name, desc)
        await m.answer(f"✅ Category “{name}” saved." + ("" if desc else " Tip: a description makes sorting far more accurate."))

    @r.message(Command("remove"))
    async def remove(m: Message, command: CommandObject):
        ok = await db.remove_category(m.from_user.id, (command.args or "").strip())
        await m.answer("🗑️ Removed." if ok else "No such category.")

    @r.message(Command("categories"))
    async def cats(m: Message):
        rows = await db.categories(m.from_user.id)
        if not rows:
            return await m.answer("No categories yet. Try: /add Work | invoices and meetings")
        await m.answer("\n".join(f"{'📌' if c['dest_chat_id'] else '▫️'} {c['name']} - {c['description'] or '(no description)'}"
                                 for c in rows) + "\n\n📌 = has a destination topic")

    @r.message(Command("bind"))
    async def bind(m: Message, command: CommandObject):
        if m.chat.type not in ("group", "supergroup"):
            return await m.answer("Send /bind <category> inside the group topic where items should go.")
        name = (command.args or "").strip()
        thread = m.message_thread_id if m.is_topic_message else None
        if await db.bind(m.from_user.id, name, m.chat.id, thread):
            await m.answer(f"📌 “{name}” items will now be copied here.")
        else:
            await m.answer("Unknown category - create it first with /add in our private chat.")

    @r.message(Command("unbind"))
    async def unbind(m: Message, command: CommandObject):
        ok = await db.bind(m.from_user.id, (command.args or "").strip(), None, None)
        await m.answer("Unbound." if ok else "Unknown category.")

    @r.message(Command("keys"))
    async def keys(m: Message):
        icon = {"ok": "🟢", "rate_limited": "🟡", "daily_quota": "🟠", "model_error": "🟣", "invalid": "🔴"}
        lines = []
        for k in deps.client.pool.status_rows():
            extra = f" - back in {k['retry_in_s']}s" if k["retry_in_s"] else ""
            lines.append(f"{icon.get(k['state'], '⚪')} {k['label']} [{k['group'][:8]}] {k['state']}{extra}"
                         f"{' ← active' if k['active'] else ''} · {k['model']}")
        await m.answer("\n".join(lines) or "No keys configured.")

    @r.message(Command("status"))
    async def status(m: Message):
        await m.answer(f"Queue: {await db.pending_count()} item(s) pending/processing.\n"
                       f"ffmpeg: {'yes' if HAVE_FFMPEG else 'NO (audio/video features limited)'}\n"
                       f"Download limit: {s.max_download_bytes // 1048576} MB"
                       f"{' (local Bot API server)' if s.local_api else ' (cloud Bot API)'}")

    @r.message(Command("recent"))
    async def recent(m: Message):
        rows = await db.recent(m.from_user.id)
        await m.answer("\n".join(f"#{x['id']} [{x['category'] or x['status']}] {x['title'] or '…'}" for x in rows)
                       or "Nothing yet.")

    @r.message(Command("get"))
    async def get(m: Message, command: CommandObject):
        try:
            item = await db.item(int((command.args or "").strip().lstrip("#")))
        except ValueError:
            item = None
        if not item or item["user_id"] != m.from_user.id:
            return await m.answer("Usage: /get <id>  (ids from /recent)")
        await bot.copy_messages(m.chat.id, item["chat_id"], json.loads(item["message_ids"]))

    # ── callbacks: change category ──
    @r.callback_query(F.data.startswith("mv:"))
    async def open_menu(cb: CallbackQuery):
        item = await db.item(int(cb.data.split(":")[1]))
        if not item or item["user_id"] != cb.from_user.id:
            return await cb.answer()
        await cb.message.edit_reply_markup(reply_markup=cats_kb(item["id"], await db.categories(item["user_id"])))
        await cb.answer()

    @r.callback_query(F.data.startswith("sv:"))
    async def set_cat(cb: CallbackQuery):
        _, sid, idx = cb.data.split(":")
        item = await db.item(int(sid))
        if not item or item["user_id"] != cb.from_user.id:
            return await cb.answer()
        rows = await db.categories(item["user_id"])
        cat = None if idx == "u" else (rows[int(idx)] if int(idx) < len(rows) else None)
        name = cat["name"] if cat else UNSORTED
        await drop_copies(bot, item["copied"])
        copied = None
        try:
            copied = await copy_to(bot, item, cat)
        except TelegramAPIError as e:
            await cb.message.answer(f"⚠️ Couldn't copy to the destination: {e}")
        await db.set_category(item["id"], name if cat else None, copied)
        await cb.message.edit_text(f"📂 {name} (set by you){' → filed' if copied else ''}\n{item['title'] or ''}",
                                   reply_markup=change_kb(item["id"]))
        await cb.answer("Moved")

    # ── button menu ──
    async def show(cb: CallbackQuery, text: str, kb: InlineKeyboardMarkup) -> None:
        try:
            await cb.message.edit_text(text[:4000], reply_markup=kb)
        except TelegramAPIError:                      # "message is not modified" etc.
            pass
        await cb.answer()

    async def cat_by_id(uid: int, idx: str) -> dict | None:
        rows = await db.categories(uid)
        return rows[int(idx)] if idx.isdigit() and int(idx) < len(rows) else None

    async def cats_screen(uid: int) -> tuple[str, InlineKeyboardMarkup]:
        rows = await db.categories(uid)
        if not rows:
            return "No categories yet. Tap ➕ to create your first one.", _kb([[("➕ Add category", "m:add")], BACK])
        btns = [(f"{'📌' if c['dest_chat_id'] else '▫️'} {c['name']}", f"m:c:{i}") for i, c in enumerate(rows)]
        grid = [btns[i:i + 2] for i in range(0, len(btns), 2)]
        return "Your categories (📌 = has a destination topic). Tap one to manage it:", _kb(grid + [[("➕ Add category", "m:add")], BACK])

    async def cat_screen(c: dict, idx: str) -> tuple[str, InlineKeyboardMarkup]:
        text = (f"📂 {c['name']}\n{c['description'] or '(no description - tap Edit description, it makes sorting much better)'}\n\n"
                f"Destination: {'a topic/group is bound ✅' if c['dest_chat_id'] else 'none (I only tag items)'}")
        rows = [[("✏️ Edit description", f"m:ed:{idx}"), ("📌 Bind to a topic", f"m:bd:{idx}")]]
        if c["dest_chat_id"]:
            rows.append([("🔓 Unbind", f"m:ub:{idx}")])
        rows.append([("🗑️ Remove", f"m:rm:{idx}"), ("⬅️ Categories", "m:cats")])
        return text, _kb(rows)

    @r.callback_query(F.data.startswith("m:"))
    async def menu_cb(cb: CallbackQuery):
        uid, parts = cb.from_user.id, cb.data.split(":")
        act = parts[1]
        if act == "skip":                                           # add flow: save without a description
            st = state.pending.pop(uid, None)
            if st and st[0] == "add_desc":
                await db.upsert_category(uid, st[1], "")
                return await show(cb, f"✅ “{st[1]}” saved without a description.",
                                  _kb([[("📂 Categories", "m:cats")], BACK]))
            return await cb.answer()
        if act not in ("add", "ed", "bd"):
            state.pending.pop(uid, None)
        if act == "home" or act == "cancel":
            return await show(cb, "Menu:", main_menu())
        if act == "help":
            return await show(cb, HELP, _kb([BACK]))
        if act == "cats":
            return await show(cb, *await cats_screen(uid))
        if act == "add":
            state.pending[uid] = ("add",)
            return await show(cb, "➕ Send me the category name (max 40 characters).", CANCEL)
        if act == "c":
            c = await cat_by_id(uid, parts[2])
            return await show(cb, *(await cat_screen(c, parts[2]) if c else await cats_screen(uid)))
        if act in ("ed", "bd", "ub", "rm", "rmy"):
            c = await cat_by_id(uid, parts[2])
            if not c:
                return await show(cb, *await cats_screen(uid))
            if act == "ed":
                state.pending[uid] = ("edit", c["name"])
                return await show(cb, f"✏️ Send the new description for “{c['name']}”.", CANCEL)
            if act == "bd":
                state.pending[uid] = ("bind", c["name"])
                return await show(cb, f"📌 Open the group topic where “{c['name']}” items should go and send any "
                                      f"message there (e.g. a dot). I must be an admin of that group.", CANCEL)
            if act == "ub":
                await db.bind(uid, c["name"], None, None)
                return await show(cb, *await cat_screen(await db.category(uid, c["name"]), parts[2]))
            if act == "rm":
                return await show(cb, f"Remove “{c['name']}”? Items already filed stay where they are.",
                                  _kb([[("🗑️ Yes, remove", f"m:rmy:{parts[2]}"), ("Cancel", f"m:c:{parts[2]}")]]))
            await db.remove_category(uid, c["name"])
            return await show(cb, *await cats_screen(uid))
        if act == "recent":
            rows = await db.recent(uid)
            if not rows:
                return await show(cb, "Nothing yet.", _kb([BACK]))
            lines = [f"#{x['id']} [{x['category'] or x['status']}] {x['title'] or '…'}" for x in rows]
            btns = [(f"↩️ #{x['id']}", f"m:get:{x['id']}") for x in rows]
            return await show(cb, "\n".join(lines) + "\n\nTap a number to get that item sent again:",
                              _kb([btns[i:i + 5] for i in range(0, len(btns), 5)] + [BACK]))
        if act == "get":
            item = await db.item(int(parts[2]))
            if not item or item["user_id"] != uid:
                return await cb.answer("Not found")
            await bot.copy_messages(cb.message.chat.id, item["chat_id"], json.loads(item["message_ids"]))
            return await cb.answer("Sent")
        if act == "status":
            return await show(cb, f"Queue: {await db.pending_count()} item(s) pending/processing.\n"
                                  f"ffmpeg: {'yes' if HAVE_FFMPEG else 'NO (audio/video features limited)'}\n"
                                  f"Download limit: {s.max_download_bytes // 1048576} MB"
                                  f"{' (local Bot API server)' if s.local_api else ' (cloud Bot API)'}",
                              _kb([[("🔄 Refresh", "m:status")], BACK]))
        if act == "keys":
            icon = {"ok": "🟢", "rate_limited": "🟡", "daily_quota": "🟠", "model_error": "🟣", "invalid": "🔴"}
            lines = []
            for k in deps.client.pool.status_rows():
                extra = f" - back in {k['retry_in_s']}s" if k["retry_in_s"] else ""
                lines.append(f"{icon.get(k['state'], '⚪')} {k['label']} [{k['group'][:8]}] {k['state']}{extra}"
                             f"{' ← active' if k['active'] else ''} · {k['model']}")
            return await show(cb, "\n".join(lines) or "No keys configured.", _kb([[("🔄 Refresh", "m:keys")], BACK]))
        await cb.answer()

    # Text replies that belong to a button flow (name/description) - must come before the generic intake.
    @r.message(F.chat.type == "private", F.text, ~F.text.startswith("/"),
               lambda m: m.from_user.id in state.pending and state.pending[m.from_user.id][0] in ("add", "add_desc", "edit"))
    async def flow_text(m: Message):
        uid, st, txt = m.from_user.id, state.pending[m.from_user.id], m.text.strip()
        if st[0] == "add":
            if not txt or len(txt) > 40 or txt.lower() == UNSORTED.lower() or "|" in txt:
                return await m.answer("Please send a name up to 40 characters (not “Unsorted”, no “|”).", reply_markup=CANCEL)
            state.pending[uid] = ("add_desc", txt)
            return await m.answer(f"Now describe what belongs in “{txt}” (your own definition - the more specific, "
                                  f"the better I sort).", reply_markup=_kb([[("Skip", "m:skip")], [("✖️ Cancel", "m:cancel")]]))
        state.pending.pop(uid)
        name = st[1]
        await db.upsert_category(uid, name, txt)
        await m.answer(f"✅ “{name}” saved.", reply_markup=_kb([[("📂 Categories", "m:cats"), ("➕ Add another", "m:add")], BACK]))

    @r.message(F.chat.type.in_({"group", "supergroup"}),
               lambda m: m.from_user is not None and m.from_user.id in state.pending and state.pending[m.from_user.id][0] == "bind")
    async def bind_by_message(m: Message):
        _, name = state.pending.pop(m.from_user.id)
        thread = m.message_thread_id if m.is_topic_message else None
        if await db.bind(m.from_user.id, name, m.chat.id, thread):
            await m.answer(f"📌 “{name}” items will now be copied here.")
        else:
            await m.answer("That category no longer exists.")

    # ── intake (private chat only; registered last so commands win) ──
    async def enqueue(msgs: list[Message]) -> None:
        msgs.sort(key=lambda x: x.message_id)
        first, uid = msgs[0], msgs[0].from_user.id
        if not await db.categories(uid):
            return await first.answer("Add at least one category first, e.g. /add Work | invoices and meetings")
        atts, ctx = [], []
        for m in msgs:
            a, c = extract(m)
            atts += a
            ctx += c
        if not atts and not ctx:
            return await first.answer("I can't read that kind of message yet.")
        if len(msgs) > 1:
            ctx.insert(0, f"This is an album of {len(msgs)} messages sent together.")
        await db.enqueue(uid, first.chat.id, [m.message_id for m in msgs],
                         {"context": ctx, "attachments": [asdict(a) for a in atts]})
        try:
            await first.react([ReactionTypeEmoji(emoji="👀")])
        except Exception:
            pass

    async def flush_album(gid: str) -> None:
        await asyncio.sleep(1.5)                    # let the rest of the album arrive
        msgs = state.albums.pop(gid, [])
        if msgs:
            await enqueue(msgs)

    @r.message(F.chat.type == "private")
    async def intake(m: Message):
        if m.text and m.text.startswith("/"):
            return await m.answer("Unknown command. /help")
        if m.media_group_id:
            bucket = state.albums.setdefault(m.media_group_id, [])
            bucket.append(m)
            if len(bucket) == 1:
                asyncio.create_task(flush_album(m.media_group_id))
            return
        await enqueue([m])

    return public, r


# ───────────────────────── entrypoint ─────────────────────────
async def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    s = Settings.from_env()
    if not s.telegram_token or not s.gemini_keys:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN and GEMINI_API_KEYS (see .env.example).")
    if not s.allowed_user_ids:
        log.warning("ALLOWED_USER_IDS is empty: the bot is open to EVERY Telegram user and spends your Gemini quota.")
    if not HAVE_FFMPEG:
        log.warning("ffmpeg/ffprobe not found: audio/video will only be handled when small enough to send raw.")
    shutil.rmtree(s.tmp_dir, ignore_errors=True)
    s.tmp_dir.mkdir(parents=True, exist_ok=True)

    session = AiohttpSession(api=TelegramAPIServer.from_base(s.api_base, is_local=s.local_api))
    bot = Bot(s.telegram_token, session=session)
    http = httpx.AsyncClient()
    pool = KeyPool(s.gemini_keys, smart_rotation=s.smart_rotation, spacing_s=s.key_min_spacing_s,
                   cooldown_s=s.key_cooldown_s, threshold=s.key_failure_threshold)
    client = GeminiClient(pool, http, primary=s.primary_model, fallback=s.fallback_model)
    db = DB(s.db_path)
    await db.open()
    deps = Deps(bot=bot, db=db, client=client, settings=s, http=http)
    state, stop = State(), asyncio.Event()

    dp = Dispatcher()
    for router in build_router(deps, state):
        dp.include_router(router)
    tasks = [asyncio.create_task(worker(deps, state, stop)) for _ in range(s.workers)]
    log.info("started: %d key(s), %d worker(s), download limit %d MB", len(pool.keys), s.workers,
             s.max_download_bytes // 1048576)
    try:
        await dp.start_polling(bot)
    finally:
        stop.set()
        for t in tasks:
            t.cancel()
        await http.aclose()
        await db.close()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
