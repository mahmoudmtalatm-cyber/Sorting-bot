from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from aiogram import Bot
from aiogram.types import Message

from .config import Settings

log = logging.getLogger(__name__)


@dataclass
class Attachment:
    kind: str                       # photo|video|video_note|animation|audio|voice|document
    file_id: str
    unique_id: str
    size: int = 0
    mime: str | None = None
    name: str | None = None
    duration: int = 0
    thumb_file_id: str | None = None
    title: str | None = None        # audio performer/title


def _thumb(obj) -> str | None:
    t = getattr(obj, "thumbnail", None)
    return t.file_id if t else None


def forward_source(m: Message) -> str | None:
    fo = getattr(m, "forward_origin", None)
    if not fo:
        return None
    try:
        if fo.type == "user":
            return fo.sender_user.full_name
        if fo.type == "hidden_user":
            return fo.sender_user_name
        if fo.type == "chat":
            return fo.sender_chat.title
        if fo.type == "channel":
            return fo.chat.title
    except AttributeError:
        pass
    return "unknown"


def extract(m: Message) -> tuple[list[Attachment], list[str]]:
    """-> (attachments that may need Gemini, plain-text context lines)."""
    atts: list[Attachment] = []
    ctx: list[str] = []
    if m.text:
        ctx.append(f"Message text: {m.text}")
    if m.caption:
        ctx.append(f"Caption: {m.caption}")
    src = forward_source(m)
    if src:
        ctx.append(f"Forwarded from: {src}")

    if m.photo:
        p = m.photo[-1]
        atts.append(Attachment("photo", p.file_id, p.file_unique_id, p.file_size or 0, "image/jpeg"))
    if m.video:
        v = m.video
        atts.append(Attachment("video", v.file_id, v.file_unique_id, v.file_size or 0, v.mime_type, v.file_name,
                               v.duration, _thumb(v)))
    if m.video_note:
        v = m.video_note
        atts.append(Attachment("video_note", v.file_id, v.file_unique_id, v.file_size or 0, "video/mp4", None,
                               v.duration, _thumb(v)))
    if m.animation:
        v = m.animation
        atts.append(Attachment("animation", v.file_id, v.file_unique_id, v.file_size or 0, v.mime_type, v.file_name,
                               v.duration, _thumb(v)))
    if m.audio:
        a = m.audio
        title = " - ".join(x for x in (a.performer, a.title) if x) or None
        atts.append(Attachment("audio", a.file_id, a.file_unique_id, a.file_size or 0, a.mime_type, a.file_name,
                               a.duration, _thumb(a), title))
    if m.voice:
        v = m.voice
        atts.append(Attachment("voice", v.file_id, v.file_unique_id, v.file_size or 0, v.mime_type or "audio/ogg",
                               None, v.duration))
    if m.document:
        d = m.document
        atts.append(Attachment("document", d.file_id, d.file_unique_id, d.file_size or 0, d.mime_type, d.file_name,
                               0, _thumb(d)))
    if m.sticker:
        s = m.sticker
        ctx.append(f"Sticker: {s.emoji or ''} (set: {s.set_name or 'n/a'})")
    if m.location:
        ctx.append(f"Location shared: {m.location.latitude:.4f}, {m.location.longitude:.4f}")
    if m.venue:
        ctx.append(f"Venue: {m.venue.title}, {m.venue.address}")
    if m.contact:
        ctx.append(f"Contact card: {m.contact.first_name} {m.contact.last_name or ''}".strip())
    if m.poll:
        ctx.append(f"Poll: {m.poll.question}")
    return atts, ctx


async def download(bot: Bot, s: Settings, file_id: str, dest: Path) -> tuple[Path, bool]:
    """
    Returns (path, owned_by_us). With a local Bot API server, getFile returns an
    absolute path on the server's disk: we read it in place (mount the server's
    data dir into this container) and never copy multi-GB files.
    """
    f = await bot.get_file(file_id)
    fp = f.file_path
    if not fp:
        raise FileNotFoundError("Telegram returned no file_path")
    if os.path.isabs(fp):
        p = Path(fp)
        if not p.exists():
            raise FileNotFoundError(
                f"{fp} not visible from the bot process - share the local Bot API server's data directory "
                f"with this process (same path)")
        return p, False
    dest.parent.mkdir(parents=True, exist_ok=True)
    await bot.download_file(fp, destination=dest)
    return dest, True
