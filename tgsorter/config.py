from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

MB = 1024 * 1024


def _bool(v: str | None, default: bool) -> bool:
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def parse_keys(raw: str) -> list[tuple[str | None, str]]:
    """'key' or 'group|key', separated by commas/whitespace -> [(group|None, key)]."""
    out, seen = [], set()
    for tok in re.split(r"[,\s]+", raw or ""):
        tok = tok.strip()
        if not tok:
            continue
        group, _, key = tok.rpartition("|")
        if key and key not in seen:
            seen.add(key)
            out.append((group or None, key))
    return out


@dataclass
class Settings:
    telegram_token: str = ""
    api_base: str = "https://api.telegram.org"
    local_api: bool = False
    allowed_user_ids: set[int] = field(default_factory=set)   # empty = everyone is allowed

    gemini_keys: list[tuple[str | None, str]] = field(default_factory=list)
    primary_model: str = "gemini-2.5-flash"
    fallback_model: str = "gemini-flash-latest"
    smart_rotation: bool = True
    key_failure_threshold: int = 2
    key_cooldown_s: float = 60.0
    key_min_spacing_s: float = 6.5

    db_path: Path = Path("data/sorter.db")
    tmp_dir: Path = Path("data/tmp")
    workers: int = 2
    confidence_threshold: float = 0.6
    max_job_attempts: int = 3

    # heavy-media knobs
    inline_max_bytes: int = 14 * MB          # raw bytes; base64 adds ~33% under the 20 MB request cap
    max_download_bytes: int = 20 * MB        # cloud Bot API getFile limit (2000 MB with a local server)
    max_analysis_seconds: int = 600          # audio/video analysed up to here
    audio_segments: int = 10                 # long audio: this many clips spread across the file (total = max_analysis_seconds)
    video_full_max_seconds: int = 90         # <= this: send the (compressed) video itself
    video_long_mode: str = "frames"          # 'frames' = audio + keyframes | 'full' = compressed video via Files API
    max_frames: int = 10
    max_attachments: int = 6                 # per message/album actually analysed
    delete_local_server_files: bool = True

    def is_allowed(self, user_id: int | None) -> bool:
        return user_id is not None and (not self.allowed_user_ids or user_id in self.allowed_user_ids)

    @classmethod
    def from_env(cls) -> "Settings":
        g = os.environ.get
        local = _bool(g("TELEGRAM_LOCAL_API"), False)
        ids = {int(x) for x in re.split(r"[,\s]+", g("ALLOWED_USER_IDS", "")) if x.strip().lstrip("-").isdigit()}
        return cls(
            telegram_token=g("TELEGRAM_BOT_TOKEN", ""),
            api_base=g("TELEGRAM_API_BASE", "https://api.telegram.org").rstrip("/"),
            local_api=local,
            allowed_user_ids=ids,
            gemini_keys=parse_keys(g("GEMINI_API_KEYS", "")),
            primary_model=g("GEMINI_PRIMARY_MODEL", "gemini-2.5-flash"),
            fallback_model=g("GEMINI_FALLBACK_MODEL", "gemini-flash-latest"),
            smart_rotation=_bool(g("SMART_ROTATION"), True),
            key_failure_threshold=int(g("KEY_FAILURE_THRESHOLD", "2")),
            key_cooldown_s=float(g("KEY_COOLDOWN_SECONDS", "60")),
            key_min_spacing_s=float(g("KEY_MIN_SPACING_SECONDS", "6.5")),
            db_path=Path(g("DB_PATH", "data/sorter.db")),
            tmp_dir=Path(g("TMP_DIR", "data/tmp")),
            workers=int(g("WORKERS", "2")),
            confidence_threshold=float(g("CONFIDENCE_THRESHOLD", "0.6")),
            max_download_bytes=(2000 if local else 20) * MB,
            max_analysis_seconds=int(g("MAX_ANALYSIS_SECONDS", "600")),
            audio_segments=int(g("AUDIO_SEGMENTS", "10")),
            video_full_max_seconds=int(g("VIDEO_FULL_MAX_SECONDS", "90")),
            video_long_mode=g("VIDEO_LONG_MODE", "frames").lower(),
            max_frames=int(g("MAX_FRAMES", "10")),
        )
