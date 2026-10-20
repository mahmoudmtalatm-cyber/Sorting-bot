"""
Gemini request layer. Python port of callGeminiWithRetry() + the Files API code
in js/gemini-uploads.js, driving the KeyPool in keypool.py.

Behaviour carried over from the web app
  * Key goes in the `x-goog-api-key` header (never the query string).
  * 401/403/API_KEY_INVALID  -> key marked invalid, rotate.
  * 429                      -> 2 strikes then rotate (see KeyPool).
  * 404 / model-ish 400      -> switch THAT key to the fallback model and strip
    temperature/topP/topK/thinkingConfig, which the fallback rejects.
  * Files-API uploads belong to one project: on rotation to a key in a different
    project the file is silently re-uploaded under the new key.
  * 5xx / network errors -> exponential back-off (2s, 4s ... capped at 30s).

Deliberate differences (a bot can't retry forever on a queue worker)
  * Attempts are bounded. When every key is excluded we raise QuotaExhausted
    carrying "retry after N seconds" and the job is re-queued, instead of a
    worker sleeping for hours.
  * A 400 that hits two DIFFERENT keys is a bad request (e.g. unreadable media),
    not two bad keys: strikes are forgiven and BadRequest is raised, so one
    poison message can't lock your whole key pool out for a cooldown.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import AsyncIterator, Callable

import httpx

from .keypool import ApiKey, KeyPool
from .media import MediaPart

log = logging.getLogger(__name__)

API = "https://generativelanguage.googleapis.com"
KEY_ERRORS = ("API_KEY_INVALID", "API_KEY_NOT_VALID", "INVALID_API_KEY", "PERMISSION_DENIED", "API key not valid")
SAMPLING_KEYS = ("temperature", "topP", "topK")


class GeminiError(Exception):
    def __init__(self, message: str, status: int = 0, data: dict | None = None):
        super().__init__(message)
        self.status, self.data = status, data or {}


class QuotaExhausted(GeminiError):
    def __init__(self, retry_after: float):
        super().__init__(f"all keys exhausted; retry in {int(retry_after)}s")
        self.retry_after = retry_after


class AllKeysInvalid(GeminiError): ...
class BadRequest(GeminiError): ...
class ContentBlocked(GeminiError): ...


def is_key_error(status: int, data: dict) -> bool:
    if status in (401, 403):
        return True
    err = (data or {}).get("error", {})
    return err.get("code") == 400 and any(k in (err.get("message") or "") for k in KEY_ERRORS)


def classify_429(data: dict) -> tuple[str, float | None]:
    """-> ('day'|'minute', retry_after_seconds|None). Heuristic on Google's error details."""
    err = (data or {}).get("error", {})
    scope, retry = "minute", None
    if re.search(r"per\s*day|daily", (err.get("message") or ""), re.I):
        scope = "day"
    for d in err.get("details", []) or []:
        t = d.get("@type", "")
        if t.endswith("RetryInfo"):
            m = re.match(r"([\d.]+)s", str(d.get("retryDelay", "")))
            if m:
                retry = float(m.group(1))
        elif t.endswith("QuotaFailure"):
            for v in d.get("violations", []) or []:
                if "perday" in ((v.get("quotaId") or "") + (v.get("quotaMetric") or "")).lower().replace("_", "").replace(" ", ""):
                    scope = "day"
    return scope, retry


async def _aiter_file(path: Path, chunk: int = 1 << 20) -> AsyncIterator[bytes]:
    with path.open("rb") as f:
        while True:
            b = await asyncio.to_thread(f.read, chunk)
            if not b:
                return
            yield b


def _check(resp: httpx.Response) -> None:
    if resp.status_code >= 400:
        try:
            data = resp.json()
        except Exception:
            data = {"error": {"message": resp.text[:300], "code": resp.status_code}}
        msg = (data.get("error") or {}).get("message") or f"HTTP {resp.status_code}"
        raise GeminiError(msg, resp.status_code, data)


class GeminiClient:
    def __init__(self, pool: KeyPool, http: httpx.AsyncClient, *, primary: str, fallback: str,
                 sleep: Callable = asyncio.sleep, max_attempts: int = 12):
        self.pool, self.http = pool, http
        self.primary, self.fallback = primary, fallback
        self._sleep, self.max_attempts = sleep, max_attempts

    # ── Files API ──────────────────────────────────────────
    async def _upload(self, part: MediaPart, key: ApiKey) -> dict:
        size = part.path.stat().st_size
        start = await self.http.post(
            f"{API}/upload/v1beta/files",
            headers={"x-goog-api-key": key.key, "X-Goog-Upload-Protocol": "resumable",
                     "X-Goog-Upload-Command": "start", "X-Goog-Upload-Header-Content-Length": str(size),
                     "X-Goog-Upload-Header-Content-Type": part.mime, "Content-Type": "application/json"},
            json={"file": {"display_name": part.path.name[:60]}}, timeout=60)
        _check(start)
        url = start.headers.get("x-goog-upload-url")
        if not url:
            raise GeminiError("Google returned no upload URL", 502)
        up = await self.http.post(
            url, headers={"Content-Length": str(size), "X-Goog-Upload-Offset": "0",
                          "X-Goog-Upload-Command": "upload, finalize"},
            content=_aiter_file(part.path), timeout=httpx.Timeout(600, connect=30))
        _check(up)
        info = up.json().get("file") or {}
        waited = 0
        while info.get("state") == "PROCESSING" and waited < 300:     # big videos/PDFs process server-side
            await self._sleep(2 if waited < 20 else 5)
            waited += 2 if waited < 20 else 5
            chk = await self.http.get(f"{API}/v1beta/{info['name']}", headers={"x-goog-api-key": key.key}, timeout=30)
            _check(chk)
            info = chk.json()
        if info.get("state") == "FAILED" or not info.get("uri"):
            raise BadRequest(f"Google failed to process {part.path.name}", 400)
        return {"name": info["name"], "uri": info["uri"], "mime": info.get("mimeType") or part.mime, "key": key.key}

    async def _gemini_parts(self, parts: list[MediaPart], key: ApiKey) -> list[dict]:
        out = []
        for p in parts:
            if p.kind == "text":
                out.append({"text": p.text})
            elif p.kind == "inline":
                out.append({"inlineData": {"mimeType": p.mime, "data": p.b64()}})
            else:
                up = p.uploads.get(key.group)            # uploads are scoped to the project
                if not up:
                    up = p.uploads[key.group] = await self._upload(p, key)
                out.append({"fileData": {"mimeType": up["mime"], "fileUri": up["uri"]}})
        return out

    async def cleanup(self, parts: list[MediaPart]) -> None:
        """Delete uploaded files now instead of waiting for the 48 h expiry (frees project storage)."""
        for p in parts:
            for up in p.uploads.values():
                try:
                    await self.http.delete(f"{API}/v1beta/{up['name']}", headers={"x-goog-api-key": up["key"]}, timeout=20)
                except Exception:
                    pass
            p.uploads.clear()

    # ── generateContent with rotation ─────────────────────
    def _body(self, parts: list[dict], system: str, schema: dict | None, low_res: bool, on_fallback: bool,
              max_tokens: int) -> dict:
        gen = {"temperature": 0.2, "maxOutputTokens": max_tokens,
               "thinkingConfig": {"thinkingBudget": 0}}
        if schema:
            gen.update(responseMimeType="application/json", responseSchema=schema)
        if low_res:
            gen["mediaResolution"] = "MEDIA_RESOLUTION_LOW"
        if on_fallback:                                    # fallback model rejects these fields
            for k in SAMPLING_KEYS:
                gen.pop(k, None)
            gen.pop("thinkingConfig", None)
        return {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": parts}], "generationConfig": gen}

    @staticmethod
    def _extract_text(data: dict) -> str:
        fb = data.get("promptFeedback") or {}
        if fb.get("blockReason"):
            raise ContentBlocked(f"blocked: {fb['blockReason']}")
        cands = data.get("candidates") or []
        if not cands:
            raise GeminiError("empty response from Gemini", 502, data)
        c = cands[0]
        txt = "".join(p.get("text", "") for p in (c.get("content") or {}).get("parts", []))
        if not txt and c.get("finishReason") in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "IMAGE_SAFETY"):
            raise ContentBlocked(f"blocked: {c['finishReason']}")
        return txt

    async def generate(self, parts: list[MediaPart], *, system: str, schema: dict | None = None,
                       low_res: bool = False, max_tokens: int = 1024,
                       on_event: Callable[[str], None] | None = None) -> str:
        pool = self.pool
        bad400: set[str] = set()
        backoff_n = 0
        for _ in range(self.max_attempts):
            key, wait = pool.acquire()
            if key is None:
                if wait == float("inf"):
                    raise AllKeysInvalid("every configured Gemini key is invalid")
                raise QuotaExhausted(wait)
            if wait:
                await self._sleep(wait)
            st = pool.state(key.id)
            model = st.model or self.primary
            try:
                gparts = await self._gemini_parts(parts, key)
                body = self._body(gparts, system, schema, low_res, bool(st.model), max_tokens)
                resp = await self.http.post(
                    f"{API}/v1beta/models/{model}:generateContent",
                    headers={"x-goog-api-key": key.key, "Content-Type": "application/json"},
                    content=json.dumps(body), timeout=httpx.Timeout(180, connect=30))
                _check(resp)
                data = resp.json()
            except GeminiError as e:
                st_code, d = e.status, e.data
                if isinstance(e, (BadRequest, ContentBlocked)):
                    raise
                if is_key_error(st_code, d):
                    pool.mark_invalid(key.id)
                    log.warning("key %s invalid -> rotating", key.label)
                    if on_event: on_event("rotated")
                    continue
                if st_code == 429:
                    scope, retry = classify_429(d)
                    tripped = pool.record_failure(key.id, 429, scope=scope, retry_after=retry)
                    log.warning("key %s hit 429 (%s)%s", key.label, scope, " -> excluded" if tripped else "")
                    if tripped:
                        if on_event: on_event("rotated")
                    else:
                        await self._sleep(min(retry or 2 * 2 ** backoff_n, 30))
                        backoff_n += 1
                    continue
                if st_code in (400, 404):
                    msg = str(e).lower()
                    if (st_code == 404 or "model" in msg) and not st.model and self.fallback != self.primary:
                        log.warning("key %s: %s rejected (%s) -> fallback model", key.label, model, st_code)
                        st.model = self.fallback            # no strike: it's a model problem, not a key problem
                        continue
                    bad400.add(key.id)
                    pool.record_failure(key.id, 400)
                    if len(bad400) >= 2 or pool.usable_count() < 2:
                        for kid in bad400:
                            pool.forgive(kid)
                        raise BadRequest(str(e), st_code, d)
                    continue
                if st_code >= 500 or st_code == 408:
                    pool.record_failure(key.id, st_code)
                    await self._sleep(min(2 * 2 ** backoff_n, 30))
                    backoff_n += 1
                    continue
                raise
            except (httpx.TransportError, asyncio.TimeoutError) as e:
                log.warning("network error on key %s: %s", key.label, type(e).__name__)
                pool.record_failure(key.id, 0)
                await self._sleep(min(2 * 2 ** backoff_n, 30))
                backoff_n += 1
                continue
            pool.record_success(key.id)
            return self._extract_text(data)
        raise GeminiError(f"gave up after {self.max_attempts} attempts")
