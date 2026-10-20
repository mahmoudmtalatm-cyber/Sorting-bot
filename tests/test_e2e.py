import json
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime
from types import SimpleNamespace

import httpx
import pytest
from aiogram.types import Chat, Message, User, Video

from tgsorter.bot import State, process_one
from tgsorter.config import Settings
from tgsorter.db import DB
from tgsorter.gemini import GeminiClient
from tgsorter.keypool import KeyPool
from tgsorter.pipeline import Deps
from tgsorter.telegram_io import Attachment, extract


def reply_json(category="Work", conf=0.9, title="Invoice", summary="ACME invoice"):
    txt = json.dumps({"category": category, "confidence": conf, "title": title, "summary": summary})
    return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": txt}]}}]})


class FakeBot:
    def __init__(self, files):
        self.files, self.sent, self.copied, self.deleted, self.downloads = files, [], [], [], []

    async def get_file(self, file_id):
        self.downloads.append(file_id)
        return SimpleNamespace(file_path=f"remote/{file_id}")

    async def download_file(self, fp, destination):
        shutil.copyfile(self.files[fp.split("/")[-1]], destination)

    async def copy_messages(self, chat_id, from_chat_id, message_ids, message_thread_id=None):
        self.copied.append((chat_id, message_thread_id, list(message_ids)))
        return [SimpleNamespace(message_id=900 + i) for i, _ in enumerate(message_ids)]

    async def delete_messages(self, chat_id, message_ids):
        self.deleted.append((chat_id, message_ids))

    async def send_message(self, chat_id, text, **kw):
        self.sent.append((chat_id, text, kw))
        return SimpleNamespace(message_id=500 + len(self.sent))


@pytest.fixture
async def env(tmp_path):
    gem_requests, responder = [], {"fn": lambda body: reply_json()}

    async def handler(req: httpx.Request):
        body = json.loads(req.content)
        gem_requests.append(body)
        return responder["fn"](body)

    s = Settings(tmp_dir=tmp_path / "tmp", key_min_spacing_s=0, max_download_bytes=20 * 1048576)
    db = DB(":memory:")
    await db.open()
    pool = KeyPool([(None, "k1"), (None, "k2")], spacing_s=0)

    async def nosleep(_): pass
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = GeminiClient(pool, http, primary=s.primary_model, fallback=s.fallback_model, sleep=nosleep)
    bot = FakeBot({})
    deps = Deps(bot=bot, db=db, client=client, settings=s, http=http)
    await db.upsert_category(1, "Work", "invoices and meetings")
    await db.upsert_category(1, "Recipes", "cooking")
    await db.bind(1, "Work", -1001, 77)                      # Work -> topic 77 of group -1001
    yield SimpleNamespace(deps=deps, db=db, bot=bot, reqs=gem_requests, responder=responder, state=State(), tmp=tmp_path)
    await db.close()


async def enqueue(env, ctx, atts=(), ids=(10,)):
    return await env.db.enqueue(1, 1, list(ids), {"context": ctx, "attachments": [asdict(a) for a in atts]})


async def test_text_message_is_classified_filed_and_acknowledged(env):
    iid = await enqueue(env, ["Message text: ACME invoice #42 due Friday"])
    assert await process_one(env.deps, env.state) is True
    assert env.bot.copied == [(-1001, 77, [10])]               # copied into the Work topic
    chat, text, kw = env.bot.sent[0]
    assert "📂 Work" in text and "filed" in text
    item = await env.db.item(iid)
    assert item["status"] == "done" and item["category"] == "Work"
    assert json.loads(item["copied"]) == {"chat_id": -1001, "ids": [900]}
    sys_prompt = env.reqs[0]["systemInstruction"]["parts"][0]["text"]
    assert "Work: invoices and meetings" in sys_prompt and "Recipes: cooking" in sys_prompt
    assert env.reqs[0]["generationConfig"]["responseSchema"]["properties"]["category"]["enum"] == ["Recipes", "Work", "Unsorted"]
    assert await process_one(env.deps, env.state) is False     # queue drained


async def test_low_confidence_is_not_filed_and_offers_category_buttons(env):
    env.responder["fn"] = lambda b: reply_json("Work", 0.3)
    await enqueue(env, ["Message text: hmm"])
    await process_one(env.deps, env.state)
    assert env.bot.copied == []
    _, text, kw = env.bot.sent[0]
    assert text.startswith("❓ Not sure")
    labels = [b.text for row in kw["reply_markup"].inline_keyboard for b in row]
    assert labels == ["Recipes", "Work", "Unsorted"]


async def test_model_returning_unknown_category_falls_back_to_unsorted(env):
    env.responder["fn"] = lambda b: reply_json("Hacked", 0.99)
    await enqueue(env, ["Message text: x"])
    await process_one(env.deps, env.state)
    assert env.bot.copied == [] and "Not sure" in env.bot.sent[0][1]


async def test_long_video_sent_as_audio_plus_frames_and_workdir_cleaned(env):
    f = env.tmp / "v.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=duration=150:size=320x240:rate=5", "-f", "lavfi", "-i", "sine=duration=150",
                    "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(f)], check=True)
    env.bot.files["vid"] = f
    a = Attachment("video", "vid", "u-vid", f.stat().st_size, "video/mp4", "lecture.mp4", 150, "th")
    await enqueue(env, ["Caption: week 3"], [a])
    await process_one(env.deps, env.state)
    parts = env.reqs[0]["contents"][0]["parts"]
    kinds = [next(iter(p)) for p in parts]
    assert kinds.count("inlineData") == 1 + 4 and "fileData" not in kinds          # 1 audio + 4 frames (150s//45+1)
    assert "mediaResolution" not in env.reqs[0]["generationConfig"]
    assert not (env.deps.settings.tmp_dir / "item1").exists()                       # temp files removed


async def test_oversized_video_never_downloaded_uses_thumbnail(env):
    thumb = env.tmp / "t.jpg"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=1:duration=1",
                    "-frames:v", "1", str(thumb)], check=True)
    env.bot.files["thumbid"] = thumb
    a = Attachment("video", "bigvid", "u-big", 900 * 1048576, "video/mp4", "wedding.mp4", 3000, "thumbid")
    await enqueue(env, [], [a])
    await process_one(env.deps, env.state)
    assert env.bot.downloads == ["thumbid"]                    # the 900 MB file was never requested
    parts = env.reqs[0]["contents"][0]["parts"]
    assert any("inlineData" in p for p in parts)
    assert "too large to download" in parts[0]["text"] and "wedding.mp4" in parts[0]["text"]


async def test_same_file_twice_skips_media_and_reuses_summary(env):
    f = env.tmp / "p.jpg"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=1:duration=1",
                    "-frames:v", "1", str(f)], check=True)
    env.bot.files["pic"] = f
    a = Attachment("photo", "pic", "u-same", f.stat().st_size, "image/jpeg")
    await enqueue(env, [], [a]); await process_one(env.deps, env.state)
    await enqueue(env, [], [a], ids=(11,)); await process_one(env.deps, env.state)
    assert env.bot.downloads == ["pic"]                        # downloaded only once
    second = env.reqs[1]["contents"][0]["parts"]
    assert len(second) == 1 and "Earlier analysis" in second[0]["text"]


async def test_quota_outage_defers_item_without_burning_attempts_and_notifies_once(env):
    env.responder["fn"] = lambda b: httpx.Response(429, json={"error": {"code": 429, "message": "quota"}})
    iid = await enqueue(env, ["Message text: a"]); await enqueue(env, ["Message text: b"], ids=(11,))
    await process_one(env.deps, env.state)
    item = await env.db.item(iid)
    assert item["status"] == "pending" and item["attempts"] == 0 and item["run_after"] > datetime.now().timestamp()
    await process_one(env.deps, env.state)
    notices = [t for _, t, _ in env.bot.sent if "rate-limited" in t]
    assert len(notices) == 1                                   # one notice per outage, not per item
    # recovery: keys come back, deferred item becomes claimable again and succeeds
    env.responder["fn"] = lambda b: reply_json()
    env.deps.client.pool.clock = lambda: datetime.now().timestamp() + 3600
    await env.db._exec("UPDATE items SET run_after=0")
    assert await process_one(env.deps, env.state) is True
    assert env.state.outage_notified is False


async def test_unexpected_failure_retries_then_offers_manual_buttons(env):
    env.responder["fn"] = lambda b: httpx.Response(400, json={"error": {"code": 400, "message": "Unable to process"}})
    iid = await enqueue(env, ["Message text: z"])
    for _ in range(3):
        await env.db._exec("UPDATE items SET run_after=0")
        await process_one(env.deps, env.state)
        env.deps.client.pool.clock = lambda: datetime.now().timestamp() + 99999   # keys recover between tries
    item = await env.db.item(iid)
    assert item["status"] == "failed" and item["attempts"] == 3
    assert "Pick a category yourself" in env.bot.sent[-1][1]


async def test_restart_requeues_items_stuck_in_processing(tmp_path):
    db = DB(tmp_path / "x.db"); await db.open()
    iid = await db.enqueue(1, 1, [1], {"context": [], "attachments": []})
    await db.claim_next()
    await db.close()
    db2 = DB(tmp_path / "x.db"); await db2.open()
    assert (await db2.item(iid))["status"] == "pending"
    await db2.close()


def test_extract_reads_aiogram_video_message_with_forward_and_caption():
    m = Message(message_id=5, date=datetime.now(), chat=Chat(id=1, type="private"), from_user=User(id=1, is_bot=False, first_name="A"),
                caption="look at this", video=Video(file_id="f", file_unique_id="u", width=640, height=360, duration=75,
                                                    file_name="clip.mp4", mime_type="video/mp4", file_size=5_000_000))
    atts, ctx = extract(m)
    assert atts[0].kind == "video" and atts[0].duration == 75 and atts[0].name == "clip.mp4"
    assert ctx == ["Caption: look at this"]
