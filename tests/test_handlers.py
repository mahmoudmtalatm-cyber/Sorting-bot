import asyncio
import json
from datetime import datetime
from types import SimpleNamespace

import httpx
import pytest
from aiogram import Bot, Dispatcher
from aiogram.types import CallbackQuery, Chat, Message, PhotoSize, Update, User

from tgsorter.bot import State, build_router
from tgsorter.config import Settings
from tgsorter.db import DB
from tgsorter.gemini import GeminiClient
from tgsorter.keypool import KeyPool
from tgsorter.pipeline import Deps

OWNER, STRANGER = 1, 2


@pytest.fixture
async def app(tmp_path):
    calls = []
    bot = Bot("123456:TESTTOKEN")

    async def fake_request(b, method, timeout=None):
        calls.append(method)
        name = type(method).__name__
        if name in ("SendMessage", "EditMessageText"):
            return Message(message_id=900 + len(calls), date=datetime.now(), chat=Chat(id=1, type="private"), text="r")
        if name == "CopyMessages":
            return [SimpleNamespace(message_id=700 + i) for i, _ in enumerate(method.message_ids)]
        return True

    bot.session.make_request = fake_request
    s = Settings(allowed_user_ids={OWNER}, tmp_dir=tmp_path)
    db = DB(":memory:"); await db.open()
    http = httpx.AsyncClient()
    deps = Deps(bot=bot, db=db, client=GeminiClient(KeyPool([("grpA", "AIzaSECRETKEY1234")], spacing_s=0), http,
                                                    primary="a", fallback="b"), settings=s, http=http)
    state = State()
    dp = Dispatcher()
    for r in build_router(deps, state):
        dp.include_router(r)
    yield SimpleNamespace(dp=dp, bot=bot, db=db, calls=calls, state=state, deps=deps)
    await http.aclose(); await db.close(); await bot.session.close()


_id = iter(range(1, 10_000))


def msg(text=None, uid=OWNER, chat_type="private", chat_id=1, thread=None, **kw):
    return Message(message_id=next(_id), date=datetime.now(), chat=Chat(id=chat_id, type=chat_type),
                   from_user=User(id=uid, is_bot=False, first_name="T"), text=text,
                   message_thread_id=thread, is_topic_message=bool(thread), **kw)


async def send(app, m):
    await app.dp.feed_update(app.bot, Update(update_id=next(_id), message=m))


def texts(app):
    return [c.text for c in app.calls if hasattr(c, "text") and c.text]


async def test_stranger_gets_their_id_on_start_and_is_ignored_otherwise(app):
    await send(app, msg("/start", uid=STRANGER))
    assert "Your Telegram ID is 2" in texts(app)[-1]
    n = len(app.calls)
    await send(app, msg("/add Hack | x", uid=STRANGER))
    await send(app, msg("some message", uid=STRANGER))
    assert len(app.calls) == n and await app.db.categories(STRANGER) == []
    assert await app.db.pending_count() == 0


async def test_add_list_remove_categories(app):
    await send(app, msg("/add Work | invoices, meetings"))
    await send(app, msg("/add Recipes"))
    await send(app, msg("/add Unsorted | nope"))
    assert [c["name"] for c in await app.db.categories(OWNER)] == ["Recipes", "Work"]
    await send(app, msg("/categories"))
    assert "Work - invoices, meetings" in texts(app)[-1]
    await send(app, msg("/remove recipes"))                      # case-insensitive
    assert [c["name"] for c in await app.db.categories(OWNER)] == ["Work"]


async def test_bind_inside_forum_topic_records_chat_and_thread(app):
    await send(app, msg("/add Work | x"))
    await send(app, msg("/bind Work", chat_type="supergroup", chat_id=-1001, thread=42))
    c = await app.db.category(OWNER, "work")
    assert (c["dest_chat_id"], c["dest_thread_id"]) == (-1001, 42)
    await send(app, msg("/bind Nope", chat_type="supergroup", chat_id=-1001, thread=42))
    assert "Unknown category" in texts(app)[-1]
    await send(app, msg("/unbind Work"))
    assert (await app.db.category(OWNER, "Work"))["dest_chat_id"] is None


async def test_message_without_categories_prompts_setup_and_queues_nothing(app):
    await send(app, msg("hello"))
    assert "Add at least one category" in texts(app)[-1] and await app.db.pending_count() == 0


async def test_plain_message_is_queued_with_context(app):
    await app.db.upsert_category(OWNER, "Work", "x")
    await send(app, msg("pay ACME 120 EUR"))
    row = await app.db.claim_next()
    assert json.loads(row["payload"]) == {"context": ["Message text: pay ACME 120 EUR"], "attachments": []}


async def test_album_is_batched_into_one_job(app):
    await app.db.upsert_category(OWNER, "Work", "x")
    for i in range(3):
        await send(app, msg(media_group_id="G1", caption="trip" if i == 0 else None,
                            photo=[PhotoSize(file_id=f"f{i}", file_unique_id=f"u{i}", width=10, height=10, file_size=100)]))
    assert await app.db.pending_count() == 0                      # still collecting
    await asyncio.sleep(1.8)
    assert await app.db.pending_count() == 1                      # one job for the whole album
    row = await app.db.claim_next()
    p = json.loads(row["payload"])
    assert len(p["attachments"]) == 3 and len(json.loads(row["message_ids"])) == 3
    assert p["context"][0].startswith("This is an album of 3")


async def test_keys_command_never_reveals_the_key(app):
    await send(app, msg("/keys"))
    out = texts(app)[-1]
    assert "...1234" in out and "SECRET" not in out and "AIza" not in out


async def test_change_category_callback_moves_copies(app):
    await app.db.upsert_category(OWNER, "Work", "x"); await app.db.upsert_category(OWNER, "Home", "y")
    await app.db.bind(OWNER, "Home", -1002, 5)
    iid = await app.db.enqueue(OWNER, 1, [10, 11], {"context": [], "attachments": []})
    await app.db.complete(iid, "Work", "t", "s", 0.9, {"chat_id": -1001, "ids": [700, 701]}, 901)
    cb = CallbackQuery(id="1", from_user=User(id=OWNER, is_bot=False, first_name="T"), chat_instance="c", data=f"sv:{iid}:0",
                       message=msg("x"))                          # idx 0 == "Home" (alphabetical)
    await app.dp.feed_update(app.bot, Update(update_id=next(_id), callback_query=cb))
    names = [type(c).__name__ for c in app.calls]
    assert "DeleteMessages" in names and "CopyMessages" in names  # old copies removed, new ones made
    item = await app.db.item(iid)
    assert item["category"] == "Home" and json.loads(item["copied"])["chat_id"] == -1002
    # someone else's item can't be touched
    other = await app.db.enqueue(99, 1, [1], {"context": [], "attachments": []})
    n = len(app.calls)
    cb2 = CallbackQuery(id="2", from_user=User(id=OWNER, is_bot=False, first_name="T"), chat_instance="c", data=f"sv:{other}:0", message=msg("x"))
    await app.dp.feed_update(app.bot, Update(update_id=next(_id), callback_query=cb2))
    assert (await app.db.item(other))["category"] is None


def last_markup(app):
    return [c for c in app.calls if hasattr(c, "reply_markup") and c.reply_markup][-1].reply_markup


def cbq(app, data, uid=OWNER):
    cb = CallbackQuery(id=str(next(_id)), from_user=User(id=uid, is_bot=False, first_name="T"), chat_instance="c",
                       data=data, message=msg("x"))
    return app.dp.feed_update(app.bot, Update(update_id=next(_id), callback_query=cb))


async def test_start_shows_button_menu(app):
    await send(app, msg("/start"))
    labels = [b.text for row in last_markup(app).inline_keyboard for b in row]
    assert "📂 Categories" in labels and "➕ Add category" in labels


async def test_add_category_entirely_with_buttons(app):
    await cbq(app, "m:add")
    await send(app, msg("Work"))
    await send(app, msg("invoices and meetings"))
    c = await app.db.category(OWNER, "Work")
    assert c["description"] == "invoices and meetings"
    await cbq(app, "m:add")
    await send(app, msg("Home"))
    await cbq(app, "m:skip")
    assert (await app.db.category(OWNER, "Home"))["description"] == ""
    assert await app.db.pending_count() == 0                    # the typed texts were not queued as items


async def test_bind_with_buttons_then_message_in_topic(app):
    await app.db.upsert_category(OWNER, "Work", "x")
    await cbq(app, "m:bd:0")
    await send(app, msg(".", chat_type="supergroup", chat_id=-1001, thread=7))
    c = await app.db.category(OWNER, "Work")
    assert (c["dest_chat_id"], c["dest_thread_id"]) == (-1001, 7)


async def test_remove_category_asks_for_confirmation(app):
    await app.db.upsert_category(OWNER, "Work", "x")
    await cbq(app, "m:rm:0")
    assert await app.db.category(OWNER, "Work")
    await cbq(app, "m:rmy:0")
    assert await app.db.category(OWNER, "Work") is None


async def test_everyone_allowed_when_no_whitelist(app):
    app.deps.settings.allowed_user_ids = set()
    await send(app, msg("/add Mine | x", uid=STRANGER))
    assert [c["name"] for c in await app.db.categories(STRANGER)] == ["Mine"]
    assert await app.db.categories(OWNER) == []                 # categories stay per-user
