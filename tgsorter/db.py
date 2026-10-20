from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS categories(
  user_id INTEGER NOT NULL,
  name TEXT NOT NULL COLLATE NOCASE,
  description TEXT NOT NULL DEFAULT '',
  dest_chat_id INTEGER,
  dest_thread_id INTEGER,
  PRIMARY KEY(user_id, name));
CREATE TABLE IF NOT EXISTS items(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  chat_id INTEGER NOT NULL,
  message_ids TEXT NOT NULL,
  payload TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',     -- pending | processing | done | failed
  attempts INTEGER NOT NULL DEFAULT 0,
  run_after REAL NOT NULL DEFAULT 0,
  category TEXT, title TEXT, summary TEXT, confidence REAL, error TEXT,
  copied TEXT,                                -- {"chat_id":..,"ids":[..]} of the copies we made
  reply_message_id INTEGER,
  created_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS items_queue ON items(status, run_after);
CREATE TABLE IF NOT EXISTS media_cache(
  unique_id TEXT PRIMARY KEY, title TEXT, summary TEXT, created_at REAL);
"""


class DB:
    def __init__(self, path: Path):
        self.path = path
        self.conn: aiosqlite.Connection | None = None
        self.lock = asyncio.Lock()

    async def open(self) -> None:
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = await aiosqlite.connect(self.path)
        self.conn.row_factory = aiosqlite.Row
        await self.conn.executescript(SCHEMA)
        # a crash mid-job must not strand work: anything 'processing' goes back to the queue
        await self.conn.execute("UPDATE items SET status='pending' WHERE status='processing'")
        await self.conn.commit()

    async def close(self) -> None:
        if self.conn:
            await self.conn.close()

    async def _all(self, sql: str, args: tuple = ()) -> list[dict]:
        async with self.conn.execute(sql, args) as cur:
            return [dict(r) for r in await cur.fetchall()]

    async def _one(self, sql: str, args: tuple = ()) -> dict | None:
        rows = await self._all(sql, args)
        return rows[0] if rows else None

    async def _exec(self, sql: str, args: tuple = ()) -> None:
        async with self.lock:
            await self.conn.execute(sql, args)
            await self.conn.commit()

    # ── categories ─────────────────────────────────────────
    async def upsert_category(self, user_id: int, name: str, description: str) -> None:
        await self._exec(
            "INSERT INTO categories(user_id,name,description) VALUES(?,?,?) "
            "ON CONFLICT(user_id,name) DO UPDATE SET description=excluded.description",
            (user_id, name, description))

    async def remove_category(self, user_id: int, name: str) -> bool:
        async with self.lock:
            cur = await self.conn.execute("DELETE FROM categories WHERE user_id=? AND name=?", (user_id, name))
            await self.conn.commit()
            return cur.rowcount > 0

    async def categories(self, user_id: int) -> list[dict]:
        return await self._all("SELECT * FROM categories WHERE user_id=? ORDER BY name", (user_id,))

    async def category(self, user_id: int, name: str) -> dict | None:
        return await self._one("SELECT * FROM categories WHERE user_id=? AND name=?", (user_id, name))

    async def bind(self, user_id: int, name: str, chat_id: int | None, thread_id: int | None) -> bool:
        async with self.lock:
            cur = await self.conn.execute(
                "UPDATE categories SET dest_chat_id=?, dest_thread_id=? WHERE user_id=? AND name=?",
                (chat_id, thread_id, user_id, name))
            await self.conn.commit()
            return cur.rowcount > 0

    # ── queue ──────────────────────────────────────────────
    async def enqueue(self, user_id: int, chat_id: int, message_ids: list[int], payload: dict) -> int:
        async with self.lock:
            cur = await self.conn.execute(
                "INSERT INTO items(user_id,chat_id,message_ids,payload,created_at) VALUES(?,?,?,?,?)",
                (user_id, chat_id, json.dumps(message_ids), json.dumps(payload), time.time()))
            await self.conn.commit()
            return cur.lastrowid

    async def claim_next(self) -> dict | None:
        async with self.lock:
            async with self.conn.execute(
                "SELECT * FROM items WHERE status='pending' AND run_after<=? ORDER BY id LIMIT 1",
                (time.time(),)) as cur:
                row = await cur.fetchone()
            if not row:
                return None
            await self.conn.execute("UPDATE items SET status='processing' WHERE id=?", (row["id"],))
            await self.conn.commit()
            return dict(row)

    async def defer(self, item_id: int, seconds: float) -> None:
        """Quota outage: back to the queue WITHOUT burning one of the item's attempts."""
        await self._exec("UPDATE items SET status='pending', run_after=? WHERE id=?", (time.time() + seconds, item_id))

    async def retry_or_fail(self, item_id: int, error: str, max_attempts: int, delay: float = 30) -> bool:
        """Returns True if the item will be retried, False if it was marked failed."""
        async with self.lock:
            async with self.conn.execute("SELECT attempts FROM items WHERE id=?", (item_id,)) as cur:
                attempts = (await cur.fetchone())["attempts"] + 1
            if attempts >= max_attempts:
                await self.conn.execute("UPDATE items SET status='failed', attempts=?, error=? WHERE id=?",
                                        (attempts, error[:500], item_id))
                retry = False
            else:
                await self.conn.execute("UPDATE items SET status='pending', attempts=?, error=?, run_after=? WHERE id=?",
                                        (attempts, error[:500], time.time() + delay * attempts, item_id))
                retry = True
            await self.conn.commit()
            return retry

    async def complete(self, item_id: int, category: str | None, title: str, summary: str, confidence: float,
                       copied: dict | None, reply_message_id: int | None) -> None:
        await self._exec(
            "UPDATE items SET status='done', category=?, title=?, summary=?, confidence=?, copied=?, "
            "reply_message_id=?, error=NULL WHERE id=?",
            (category, title, summary, confidence, json.dumps(copied) if copied else None, reply_message_id, item_id))

    async def set_category(self, item_id: int, category: str | None, copied: dict | None) -> None:
        await self._exec("UPDATE items SET category=?, copied=? WHERE id=?",
                         (category, json.dumps(copied) if copied else None, item_id))

    async def item(self, item_id: int) -> dict | None:
        return await self._one("SELECT * FROM items WHERE id=?", (item_id,))

    async def recent(self, user_id: int, limit: int = 10) -> list[dict]:
        return await self._all(
            "SELECT id,category,title,status,created_at FROM items WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, limit))

    async def pending_count(self) -> int:
        row = await self._one("SELECT COUNT(*) AS n FROM items WHERE status IN ('pending','processing')")
        return row["n"]

    # ── media cache ────────────────────────────────────────
    async def cached_media(self, unique_id: str) -> dict | None:
        return await self._one("SELECT * FROM media_cache WHERE unique_id=?", (unique_id,))

    async def cache_media(self, unique_id: str, title: str, summary: str) -> None:
        await self._exec("INSERT OR REPLACE INTO media_cache VALUES(?,?,?,?)", (unique_id, title, summary, time.time()))
