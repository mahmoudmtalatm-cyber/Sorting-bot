from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import httpx
from aiogram import Bot

from .classifier import Classification, classify
from .config import Settings
from .db import DB
from .gemini import GeminiClient
from .media import MediaPart, Prepared, prepare_attachment, rebalance_inline, text_part
from .telegram_io import Attachment, download

log = logging.getLogger(__name__)


@dataclass
class Deps:
    bot: Bot
    db: DB
    client: GeminiClient
    settings: Settings
    http: httpx.AsyncClient


async def _fetch(deps: Deps, att: Attachment, workdir: Path, trash: list[Path]) -> tuple[Path | None, Path | None]:
    """Download the file (or just its thumbnail when it can't be fetched). Never raises for 'too big'."""
    s = deps.settings
    src: Path | None = None
    if att.size and att.size > s.max_download_bytes:
        log.info("%s %s MB > limit %s MB: metadata+thumbnail only", att.kind, att.size // 1048576,
                 s.max_download_bytes // 1048576)
    else:
        try:
            src, ours = await download(deps.bot, s, att.file_id, workdir / f"{att.unique_id}.src")
            if not ours and s.delete_local_server_files:
                trash.append(src)                      # the local server's copy; safe to drop afterwards
        except Exception as e:                         # e.g. "file is too big" when size was unknown
            log.warning("download failed for %s: %s", att.kind, e)
    thumb = None
    if src is None and att.thumb_file_id:
        try:
            thumb, _ = await download(deps.bot, s, att.thumb_file_id, workdir / f"{att.unique_id}.thumb")
        except Exception as e:
            log.warning("thumbnail download failed: %s", e)
    return src, thumb


async def analyze(item: dict, deps: Deps) -> Classification:
    s, db = deps.settings, deps.db
    payload = json.loads(item["payload"])
    cats = await db.categories(item["user_id"])
    atts = [Attachment(**a) for a in payload["attachments"]]
    context: list[str] = payload["context"]
    workdir = s.tmp_dir / f"item{item['id']}"
    trash: list[Path] = []
    parts: list[MediaPart] = []
    try:
        notes: list[str] = []
        low_res = False
        cached = await db.cached_media(atts[0].unique_id) if len(atts) == 1 else None
        if cached:
            # Same file seen before: skip download, ffmpeg and the heavy Gemini call entirely.
            context = context + [f"Earlier analysis of this exact file - title: {cached['title']}; "
                                 f"summary: {cached['summary']}"]
            notes.append("The attachment itself is not re-sent; rely on the earlier analysis above.")
        else:
            for att in atts[: s.max_attachments]:
                src, thumb = await _fetch(deps, att, workdir, trash)
                pr: Prepared = await prepare_attachment(att, src, thumb, workdir, s)
                parts += pr.parts
                notes += pr.notes
                low_res = low_res or pr.low_res
            if len(atts) > s.max_attachments:
                notes.append(f"{len(atts) - s.max_attachments} more attachment(s) in this album were not analysed.")
            rebalance_inline(parts, s.inline_max_bytes)

        result = await classify(deps.client, cats, context, notes, parts, low_res=low_res)
        if len(atts) == 1 and not cached and result.summary:
            await db.cache_media(atts[0].unique_id, result.title, result.summary)
        return result
    finally:
        await deps.client.cleanup(parts)               # delete Gemini-side uploads now, don't wait 48 h
        shutil.rmtree(workdir, ignore_errors=True)
        for p in trash:
            p.unlink(missing_ok=True)
