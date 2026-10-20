"""
Turns a downloaded Telegram file into the cheapest set of Gemini "parts" that
still lets the model sort it correctly.

Why this exists (numbers from Google's docs): audio costs ~32 tokens/s, video
~300 tokens/s (~100 at low media resolution), and the Files API samples video at
1 fps with audio at ~1 kbps anyway. So for *sorting* we never need the original:

  image      -> downscale to <=1568 px JPEG                      (inline)
  voice/audio-> mono 16 kHz 32 kbps MP3; if longer than N s, N s total as clips
                spread across the whole file, each labelled with its timestamps (inline)
  short video-> 360p / 2 fps / CRF 32 H.264, LOW media resolution (inline)
  long video -> audio track (clips across the file) + up to K keyframes spread across the
                WHOLE video, as inline JPEGs. ~10x fewer tokens than the video.
  PDF        -> as-is (inline if small, Files API if big)
  text file  -> first 20 KB as text
  other      -> metadata only (name, mime, size, caption)

Anything that ends up above the inline limit goes through the Files API (handled
by gemini.py, including re-upload when a key rotation changes project).
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .config import Settings

log = logging.getLogger(__name__)

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))

TEXT_EXT = {".txt", ".md", ".csv", ".tsv", ".json", ".xml", ".html", ".log", ".py", ".js", ".ts",
            ".java", ".c", ".cpp", ".go", ".rs", ".sql", ".yaml", ".yml", ".ini", ".toml", ".srt", ".vtt"}
GEMINI_IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"}


# ───────────────────────── parts ─────────────────────────
@dataclass
class MediaPart:
    """One piece of a Gemini request. kind: 'text' | 'inline' | 'file'."""
    kind: str
    text: str | None = None
    mime: str | None = None
    path: Path | None = None
    uploads: dict = field(default_factory=dict)   # project group -> {name, uri, mime, key}
    _b64: str | None = None

    def size(self) -> int:
        return self.path.stat().st_size if self.path else 0

    def b64(self) -> str:
        if self._b64 is None:
            self._b64 = base64.b64encode(self.path.read_bytes()).decode()
        return self._b64


def text_part(t: str) -> MediaPart:
    return MediaPart("text", text=t)


def file_part(path: Path, mime: str, inline_max: int) -> MediaPart:
    return MediaPart("inline" if path.stat().st_size <= inline_max else "file", mime=mime, path=path)


def rebalance_inline(parts: list[MediaPart], inline_max: int) -> None:
    """The ~20 MB inline cap is for the WHOLE request; push the biggest parts to the Files API."""
    inl = [p for p in parts if p.kind == "inline"]
    total = sum(p.size() for p in inl)
    for p in sorted(inl, key=lambda x: x.size(), reverse=True):
        if total <= inline_max:
            break
        p.kind, total = "file", total - p.size()


# ───────────────────────── ffmpeg helpers ─────────────────────────
async def _run(*cmd: str, timeout: float = 900) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, "timeout"
    return proc.returncode, out.decode(errors="replace")


async def _ffmpeg(*args: str, timeout: float = 900) -> bool:
    rc, out = await _run("ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *args, timeout=timeout)
    if rc != 0:
        log.warning("ffmpeg failed (%s): %s", rc, out[-300:])
    return rc == 0


@dataclass
class Probe:
    duration: float = 0.0
    has_audio: bool = False
    has_video: bool = False


async def probe(path: Path) -> Probe:
    if not HAVE_FFMPEG:
        return Probe()
    rc, out = await _run("ffprobe", "-v", "error", "-print_format", "json",
                         "-show_format", "-show_streams", str(path), timeout=60)
    if rc != 0:
        return Probe()
    try:
        j = json.loads(out)
    except json.JSONDecodeError:
        return Probe()
    streams = j.get("streams", [])
    dur = 0.0
    try:
        dur = float(j.get("format", {}).get("duration") or 0)
    except ValueError:
        pass
    if not dur:
        for s in streams:
            try:
                dur = max(dur, float(s.get("duration") or 0))
            except ValueError:
                pass
    return Probe(
        duration=dur,
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        has_video=any(s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")
                      for s in streams),
    )


def fmt_ts(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ───────────────────────── prepared result ─────────────────────────
@dataclass
class Prepared:
    parts: list[MediaPart] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)   # told to the model so it knows what it is (not) seeing
    low_res: bool = False                            # ask Gemini for MEDIA_RESOLUTION_LOW


# ───────────────────────── per-type preparation ─────────────────────────
async def _prep_image(src: Path, out: Path, mime: str | None, s: Settings) -> Prepared:
    pr = Prepared()
    if HAVE_FFMPEG:
        jpg = out.with_suffix(".jpg")
        if await _ffmpeg("-i", str(src), "-frames:v", "1", "-vf", "scale='min(1568,iw)':-2",
                         "-q:v", "3", str(jpg)):
            pr.parts.append(file_part(jpg, "image/jpeg", s.inline_max_bytes))
            return pr
    if mime in GEMINI_IMAGE_MIMES and src.stat().st_size <= s.inline_max_bytes:
        pr.parts.append(file_part(src, mime, s.inline_max_bytes))
    else:
        pr.notes.append("Image could not be decoded; classify from caption/metadata only.")
    return pr


async def _prep_audio(src: Path, out: Path, info: Probe, mime: str | None, s: Settings) -> Prepared:
    pr = Prepared()
    if not HAVE_FFMPEG:
        if src.stat().st_size <= s.inline_max_bytes and mime and mime.startswith("audio/"):
            pr.parts.append(file_part(src, mime, s.inline_max_bytes))
        else:
            pr.notes.append("Audio not analysed (ffmpeg unavailable and file too large).")
        return pr
    lim = s.max_analysis_seconds
    dur = info.duration
    n = max(1, s.audio_segments)
    if dur and dur > lim and n > 1:
        # Long audio: same total amount of audio (lim seconds), but taken as n clips spread evenly over
        # the WHOLE recording instead of only the first lim seconds. Each clip is labelled with its real
        # timestamps so the model can point back to where in the file things are said.
        seg = lim / n
        slice_len = dur / n
        done = 0
        for i in range(n):
            start = i * slice_len + (slice_len - seg) / 2      # clip centred in its slice
            end = start + seg
            mp3 = out.parent / f"{out.name}.seg{i:02d}.mp3"
            if not await _ffmpeg("-ss", f"{start:.2f}", "-i", str(src), "-t", f"{seg:.2f}", "-vn", "-ac", "1",
                                 "-ar", "16000", "-c:a", "libmp3lame", "-b:a", "32k", str(mp3)) or not mp3.exists():
                continue
            pr.parts.append(text_part(f"[audio clip {i + 1}/{n}: {fmt_ts(start)} - {fmt_ts(end)} of the original]"))
            pr.parts.append(file_part(mp3, "audio/mp3", s.inline_max_bytes))
            done += 1
        if done:
            pr.notes.append(
                f"Audio is {fmt_ts(dur)} long. You are given {done} clips of {fmt_ts(seg)} each ({fmt_ts(seg * done)} "
                f"total), spread evenly across the whole recording; each clip is preceded by its time range. "
                f"The parts between clips were not heard.")
            return pr
        pr.parts.clear()                                       # all clip extractions failed -> fall back below
    mp3 = out.with_suffix(".mp3")
    ok = await _ffmpeg("-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame",
                       "-b:a", "32k", "-t", str(lim), str(mp3))
    if not ok:
        pr.notes.append("Audio could not be decoded; classify from caption/metadata only.")
        return pr
    pr.parts.append(file_part(mp3, "audio/mp3", s.inline_max_bytes))
    if dur and dur > lim:
        pr.notes.append(f"Audio is {fmt_ts(dur)} long; only the first {fmt_ts(lim)} was analysed.")
    else:
        pr.notes.append(f"Audio is {fmt_ts(dur)} long; the whole recording is provided." if dur else
                        "Audio duration unknown.")
    return pr


async def _frames(src: Path, out_dir: Path, dur: float, n: int, s: Settings) -> list[tuple[float, MediaPart]]:
    res = []
    for i in range(n):
        t = dur * (i + 0.5) / n          # centre of each of n equal slices
        f = out_dir / f"frame_{i:02d}.jpg"
        if await _ffmpeg("-ss", f"{t:.2f}", "-i", str(src), "-frames:v", "1",
                         "-vf", "scale='min(768,iw)':-2", "-q:v", "4", str(f), timeout=120) and f.exists():
            res.append((t, file_part(f, "image/jpeg", s.inline_max_bytes)))
    return res


async def _prep_video(src: Path, out: Path, info: Probe, s: Settings) -> Prepared:
    pr = Prepared()
    if not HAVE_FFMPEG or not info.duration:
        if HAVE_FFMPEG is False and src.stat().st_size <= s.inline_max_bytes:
            pr.parts.append(file_part(src, "video/mp4", s.inline_max_bytes))
            pr.low_res = True
        else:
            pr.notes.append("Video not analysed (could not read it); classify from caption/metadata only.")
        return pr

    dur, lim = info.duration, s.max_analysis_seconds
    short = dur <= s.video_full_max_seconds

    if short or s.video_long_mode == "full":
        mp4 = out.with_suffix(".mp4")
        args = ["-i", str(src), "-t", str(lim), "-map", "0:v:0", "-vf", "scale=-2:'min(360,ih)',fps=2",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "32", "-pix_fmt", "yuv420p"]
        args += ["-map", "0:a:0?", "-c:a", "aac", "-b:a", "24k", "-ac", "1", "-ar", "16000"] if info.has_audio else ["-an"]
        args += ["-movflags", "+faststart", str(mp4)]
        if await _ffmpeg(*args, timeout=1800):
            pr.parts.append(file_part(mp4, "video/mp4", s.inline_max_bytes))
            pr.low_res = True
            if dur > lim:
                pr.notes.append(f"Video is {fmt_ts(dur)} long; only the first {fmt_ts(lim)} was analysed.")
            return pr
        # transcode failed -> fall through to the frames path

    # long video: audio track + keyframes across the whole video
    n = max(3, min(s.max_frames, int(dur // 45) + 1))
    if info.has_audio:
        a = await _prep_audio(src, out, info, None, s)
        pr.parts += a.parts
        pr.notes += a.notes
    fr = await _frames(src, out.parent, dur, n, s)
    for t, p in fr:
        pr.parts.append(text_part(f"[video frame at {fmt_ts(t)}]"))
        pr.parts.append(p)
    pr.notes.append(f"Video is {fmt_ts(dur)} long; you are given its audio track and {len(fr)} still frames "
                    f"sampled evenly across the whole video instead of the full video.")
    return pr


def _read_text(path: Path, limit: int = 20_000) -> str:
    with path.open("rb") as f:
        return f.read(limit).decode("utf-8", errors="replace")


async def prepare_attachment(att, src: Path | None, thumb: Path | None, workdir: Path, s: Settings) -> Prepared:
    """
    att: telegram_io.Attachment. src=None means the file could not be downloaded
    (too big for the cloud Bot API, or the download failed) -> degrade to the
    Telegram-provided thumbnail + metadata instead of failing.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    out = workdir / f"{att.unique_id or 'file'}"
    pr = Prepared()
    meta = [f"kind={att.kind}"]
    if att.name:
        meta.append(f"filename={att.name}")
    if att.mime:
        meta.append(f"mime={att.mime}")
    if att.size:
        meta.append(f"size={att.size / 1048576:.1f}MB")
    if att.duration:
        meta.append(f"duration={fmt_ts(att.duration)}")
    if att.title:
        meta.append(f"title={att.title}")
    pr.notes.append("Attachment metadata: " + ", ".join(meta))

    if src is None:
        pr.notes.append("The file itself was too large to download; judge from the metadata"
                        + (" and the thumbnail." if thumb else "."))
        if thumb:
            tp = await _prep_image(thumb, workdir / "thumb", "image/jpeg", s)
            pr.parts += tp.parts
        return pr

    kind, mime, ext = att.kind, (att.mime or ""), Path(att.name or "").suffix.lower()
    # documents that are really media
    if kind == "document":
        if mime.startswith("image/"):
            kind = "photo"
        elif mime.startswith("audio/"):
            kind = "audio"
        elif mime.startswith("video/"):
            kind = "video"

    if kind == "photo":
        sub = await _prep_image(src, out, mime, s)
    elif kind in ("voice", "audio"):
        sub = await _prep_audio(src, out, await probe(src), mime, s)
    elif kind in ("video", "video_note", "animation"):
        sub = await _prep_video(src, out, await probe(src), s)
    elif kind == "document" and (mime == "application/pdf" or ext == ".pdf"):
        sub = Prepared(parts=[file_part(src, "application/pdf", s.inline_max_bytes)])
    elif kind == "document" and (mime.startswith("text/") or ext in TEXT_EXT):
        sub = Prepared(parts=[text_part(f"[contents of {att.name or 'file'}, first 20 KB]\n{_read_text(src)}")])
    else:
        sub = Prepared(notes=["Binary/unsupported document type: classify from filename, mime and caption."])
    pr.parts += sub.parts
    pr.notes += sub.notes
    pr.low_res = sub.low_res
    return pr
