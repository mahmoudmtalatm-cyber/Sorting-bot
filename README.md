# Telegram sorter bot (Gemini, free tier, key rotation)

Send or forward anything to the bot (text, photos, voice, audio, video, PDFs, albums). Gemini
reads it, picks one of **your** categories, and the bot copies the message into the matching
topic of your own group and tags it in a local database. Wrong guess? Tap **✏️ Change category**.

```
Telegram ──► handler ──► SQLite queue ──► worker ──► download ──► ffmpeg shrink ──► Gemini (key pool)
 (instant ack 👀)        (survives restarts)                                            │
                                                  copy_messages into topic ◄── category + title + summary
```

## Setup

```bash
sudo apt install ffmpeg            # or brew install ffmpeg   (needed for audio/video)
pip install -r requirements.txt
cp .env.example .env               # fill TELEGRAM_BOT_TOKEN, GEMINI_API_KEYS (ALLOWED_USER_IDS optional)
python -m tgsorter.bot
```

1. Message the bot `/start` – if you're not whitelisted it shows your numeric ID; put it in `ALLOWED_USER_IDS`.
   Everything else is **buttons** (tap **☰ Menu** any time):
   * **📂 Categories → ➕ Add category** – send a name, then a description (your own definition – the more specific, the better the sorting).
     Tap a category to edit its description, bind/unbind a topic, or remove it.
   * **📌 Bind to a topic** – make a group, enable **Topics**, add the bot as admin, tap *Bind to a topic* on a category, then send any
     message inside the topic you want. Sorted items are *copied* there (server-side, no re-upload). Skip it and the bot only tags + remembers.
   * **🕘 Recent** (tap a number to get the item again), **📊 Status**, **🔑 Keys**.
2. Forward/send things to the bot in private chat. The old slash commands still work as shortcuts.

## Heavy media: how it's handled

Gemini bills by tokens, and the free tier is limited by requests *and* tokens per minute/day. For sorting
you never need the original quality, so the bot shrinks before sending (token costs per Google's docs:
audio ≈ 32 tok/s, video ≈ 300 tok/s, ≈ 100 at low media resolution):

| Input | What is sent to Gemini | Why |
|---|---|---|
| Image | ≤1568 px JPEG | smaller upload, same understanding |
| Voice / audio | mono 16 kHz 32 kbps MP3. Up to `MAX_ANALYSIS_SECONDS` (600) it's sent whole; longer files are sent as `AUDIO_SEGMENTS` (10) clips **spread evenly over the whole recording** (same 600 s total), each labelled with its real timestamps | ~4 KB/s, fits inline. The reply lists the time ranges (e.g. `12:05-13:05 …`) that support the category |
| Video ≤ 90 s | 360p / 2 fps H.264 with `MEDIA_RESOLUTION_LOW` | ~3× fewer tokens than default |
| Video > 90 s | audio track **(clips spread across the file, as above) + up to 10 keyframes spread across the whole video** | ≈ 10× fewer tokens than the video; nothing needs the Files API |
| `VIDEO_LONG_MODE=full` | compressed video via Files API | when what's *shown* matters more than what's said |
| PDF | as-is (inline ≤ 14 MB, else Files API) | native support |
| Text files | first 20 KB as text | |
| Other / unreadable | filename + mime + size + caption | graceful fallback |

Other safeguards
* **Telegram's 20 MB download cap** (cloud Bot API). Bigger files are **not downloaded**: the bot classifies from
  the metadata (name, duration, size, caption) plus Telegram's own thumbnail, and says so in the prompt so the
  model lowers its confidence. For full analysis of large files run a local Bot API server (below).
* **Files API** (2 GB/file, 20 GB/project, files auto-expire after 48 h) is only used when something exceeds the
  inline limit. Uploaded files are **deleted right after use**, and re-uploaded automatically if rotation moves
  the request to a key in a different project (uploads are project-scoped).
* **Never re-analyse the same file**: results are cached by Telegram's `file_unique_id`; a re-forward costs one tiny text call.
* **Albums** are collected for 1.5 s and sorted as *one* job (one category for all, ≤ 6 attachments analysed).
* **Nothing blocks the handler**: messages are queued in SQLite and processed by `WORKERS` async workers; the queue survives restarts.
* **Disk**: every job has its own temp dir, removed in a `finally`.

### Local Bot API server (files up to 2 GB)

```yaml
# docker-compose.yml
services:
  tg-api:
    image: aiogram/telegram-bot-api:latest   # community image; or build tdlib/telegram-bot-api yourself
    environment: { TELEGRAM_API_ID: "…", TELEGRAM_API_HASH: "…", TELEGRAM_LOCAL: "1" }   # from my.telegram.org
    volumes: [ "tgdata:/var/lib/telegram-bot-api" ]
    ports: [ "127.0.0.1:8081:8081" ]
volumes: { tgdata: }
```
Then set `TELEGRAM_API_BASE=http://localhost:8081` and `TELEGRAM_LOCAL_API=1`. Requirements: (1) call `logOut` on the
cloud API once before switching a bot to a local server, (2) the bot process must see the server's data directory at the
**same path** (the server returns absolute paths; the bot reads them in place and deletes its copy afterwards).

## Key rotation (ported from `api-rotation.js` / `callGeminiWithRetry`)

Kept as-is: `x-goog-api-key` header · 2 consecutive 429s (or plain 400s) exclude a key · 60 s cooldown with lazy
recovery · walk forward 1→2→3→1 preferring healthy keys · 401/403/`API_KEY_INVALID` marks a key invalid ·
`gemini-2.5-flash` → `gemini-flash-latest` fallback (stripping `temperature/topP/topK/thinkingConfig`) ·
file re-upload on rotation · `SMART_ROTATION=0` switch.

Changed, and why:
* **Quotas are per Google Cloud project, not per key** (Google's docs). Two keys made in the same project share one
  quota, so rotating between them gains nothing. Tag them: `GEMINI_API_KEYS=projA|key1,projA|key2,projB|key3`.
* **Daily quota ≠ per-minute limit.** A 429 mentioning "per day" excludes the whole project until midnight
  Pacific (when Google resets it) instead of a useless 60 s cooldown; a `retryDelay` from Google extends the cooldown.
* **Per-key pacing** (`KEY_MIN_SPACING_SECONDS`, 6.5 s ≈ 9 req/min like the web app) instead of one global gate, so
  three healthy keys ≈ 3× throughput.
* **Bounded, not infinite.** When every key is excluded the item goes back on the queue with a "retry at HH:MM" and
  the user gets one notice per outage — workers never sleep for hours.
* **A poison message can't lock your keys.** A 400 that hits two *different* keys is treated as a bad request
  (e.g. unreadable media): strikes are forgiven and only that item fails (3 attempts, then you get manual category buttons).
  A 400 only triggers the model fallback if it mentions the model (or is a 404).

`/keys` shows each key's state (masked – `...1234`), project group, and time until recovery. Keys are never logged.

## Security notes
* By default **anyone** can use the bot (`ALLOWED_USER_IDS` empty); each user has their own categories and history. It spends **your** quota, so set `ALLOWED_USER_IDS=id1,id2` to restrict it - everyone else is then ignored.
* Message content is passed to the model as data, with an instruction not to follow instructions inside it; the
  model's category is validated against your list (an invented category becomes "Unsorted").
* Low confidence (< `CONFIDENCE_THRESHOLD`, 0.6) is **not** auto-filed; you get buttons instead.

## Tests
`pip install -r requirements-dev.txt && pytest` – 56 tests: rotation state machine, Gemini retry/rotate/re-upload
against a mocked API, real ffmpeg on synthetic video/audio/images, queue + delivery with a fake Telegram, command
handlers and access control through aiogram's dispatcher. Not covered (needs real credentials): live Telegram and live Gemini.

## Known limits / easy extensions
* Audio between the sampled clips is not heard; the model is told so and only cites ranges it was given.
* Free-tier numbers (RPM/TPM/RPD, model names) change – check your limits in AI Studio; everything is env-configurable.
* Re-sorting everything after you change categories can reuse the stored summaries (text-only calls); not wired to a command yet.
