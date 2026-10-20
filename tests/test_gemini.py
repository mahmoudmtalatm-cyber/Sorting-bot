import json

import httpx
import pytest

from tgsorter.gemini import (AllKeysInvalid, BadRequest, GeminiClient, QuotaExhausted, classify_429)
from tgsorter.keypool import KeyPool
from tgsorter.media import MediaPart, text_part

OK = {"candidates": [{"content": {"parts": [{"text": '{"category":"Work","confidence":0.9,"title":"t","summary":"s"}'}]}}]}


def err(code, msg="x", details=None):
    return httpx.Response(code, json={"error": {"code": code, "message": msg, "details": details or []}})


class Fake:
    """Programmable fake of generativelanguage.googleapis.com, keyed by x-goog-api-key."""
    def __init__(self, script):
        self.script, self.calls, self.uploads, self.deleted = script, [], [], []

    async def __call__(self, req: httpx.Request):
        key = req.headers.get("x-goog-api-key")
        path = req.url.path
        if path.endswith("/upload/v1beta/files"):
            return httpx.Response(200, headers={"x-goog-upload-url": "https://up.example/session"})
        if req.url.host == "up.example":
            await req.aread()
            self.uploads.append(key_of_session[0])
            return httpx.Response(200, json={"file": {"name": f"files/{key_of_session[0]}", "uri": f"uri-{key_of_session[0]}",
                                                      "state": "ACTIVE", "mimeType": "video/mp4"}})
        if req.method == "DELETE":
            self.deleted.append(path)
            return httpx.Response(200, json={})
        body = json.loads(req.content)
        self.calls.append((key, path, body))
        step = self.script[key].pop(0) if self.script.get(key) else OK_RESP
        return step(body) if callable(step) else step


OK_RESP = httpx.Response(200, json=OK)
key_of_session = [None]


def client(script, keys=(("g1", "k1"), ("g2", "k2")), **kw):
    fake = Fake(script)
    http = httpx.AsyncClient(transport=httpx.MockTransport(fake))
    pool = KeyPool(list(keys), spacing_s=0)
    async def nosleep(_): pass
    return GeminiClient(pool, http, primary="gemini-2.5-flash", fallback="gemini-flash-latest", sleep=nosleep, **kw), fake, pool


async def test_happy_path_uses_header_not_query_and_json_schema():
    c, fake, _ = client({})
    out = await c.generate([text_part("hi")], system="sys", schema={"type": "OBJECT"})
    assert '"Work"' in out
    key, path, body = fake.calls[0]
    assert key == "k1" and "key=" not in path
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}


async def test_two_429s_rotate_to_next_key_and_succeed():
    c, fake, pool = client({"k1": [err(429), err(429)]})
    await c.generate([text_part("hi")], system="s")
    assert [k for k, *_ in fake.calls] == ["k1", "k1", "k2"]
    assert pool.is_excluded(pool.keys[0].id) and pool.active_id == pool.keys[1].id


async def test_daily_quota_429_excludes_immediately_no_second_strike():
    day = [{"@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]
    c, fake, pool = client({"k1": [err(429, "quota", day)]})
    await c.generate([text_part("hi")], system="s")
    assert [k for k, *_ in fake.calls] == ["k1", "k2"]
    assert pool.state(pool.keys[0].id).reason == "daily_quota"


def test_classify_429_parses_retry_delay_and_scope():
    d = {"error": {"message": "slow down", "details": [{"@type": "x.RetryInfo", "retryDelay": "34s"}]}}
    assert classify_429(d) == ("minute", 34.0)
    assert classify_429({"error": {"message": "Quota exceeded: requests per day"}})[0] == "day"


async def test_all_keys_exhausted_raises_quota_exhausted_with_wait():
    c, fake, _ = client({"k1": [err(429)] * 2, "k2": [err(429)] * 2})
    with pytest.raises(QuotaExhausted) as e:
        await c.generate([text_part("hi")], system="s")
    assert 0 < e.value.retry_after <= 60


async def test_invalid_key_rotates_and_all_invalid_raises():
    c, fake, pool = client({"k1": [err(403, "denied")]})
    await c.generate([text_part("hi")], system="s")
    assert pool.state(pool.keys[0].id).invalid
    c2, _, _ = client({"k1": [err(400, "API key not valid")], "k2": [err(401)]})
    with pytest.raises(AllKeysInvalid):
        await c2.generate([text_part("hi")], system="s")


async def test_404_switches_that_key_to_fallback_and_strips_params():
    c, fake, pool = client({"k1": [err(404, "model not found")]}, keys=(("g", "k1"),))
    await c.generate([text_part("hi")], system="s")
    (_, p1, b1), (_, p2, b2) = fake.calls
    assert "gemini-2.5-flash:" in p1 and "gemini-flash-latest:" in p2
    assert "thinkingConfig" in b1["generationConfig"] and "thinkingConfig" not in b2["generationConfig"]
    assert "temperature" not in b2["generationConfig"]
    assert pool.state(pool.keys[0].id).excluded_until == 0      # no strike for a model problem


async def test_poison_400_on_two_keys_raises_bad_request_and_forgives_keys():
    c, fake, pool = client({"k1": [err(400, "Unable to process input image")] * 3,
                            "k2": [err(400, "Unable to process input image")] * 3})
    with pytest.raises(BadRequest):
        await c.generate([text_part("hi")], system="s")
    assert not pool.all_excluded()                                  # one bad message must not lock the pool
    # same 2-strike semantics as the web app (k1, k1) then the second distinct key proves it's the request
    assert [k for k, *_ in fake.calls] == ["k1", "k1", "k2"]


async def test_500_backs_off_then_retries():
    sleeps = []
    c, fake, _ = client({"k1": [err(503)]})
    async def rec(s): sleeps.append(s)
    c._sleep = rec
    await c.generate([text_part("hi")], system="s")
    assert sleeps and sleeps[0] == 2


async def test_files_api_reupload_under_new_project_after_rotation(tmp_path):
    big = tmp_path / "v.mp4"; big.write_bytes(b"x" * 100)
    part = MediaPart("file", mime="video/mp4", path=big)
    c, fake, pool = client({"k1": [err(429), err(429)]})
    # the fake needs to know which key owns an upload session: patch via start-request header capture
    orig = fake.__call__
    async def tracking(req):
        if req.url.path.endswith("/upload/v1beta/files"):
            key_of_session[0] = req.headers["x-goog-api-key"]
        return await orig(req)
    c.http = httpx.AsyncClient(transport=httpx.MockTransport(tracking))
    await c.generate([text_part("hi"), part], system="s", low_res=True)
    assert fake.uploads == ["k1", "k2"]                              # uploaded under k1, re-uploaded under k2
    last = fake.calls[-1]
    assert last[0] == "k2" and last[2]["contents"][0]["parts"][1]["fileData"]["fileUri"] == "uri-k2"
    assert last[2]["generationConfig"]["mediaResolution"] == "MEDIA_RESOLUTION_LOW"
    await c.cleanup([part])
    assert len(fake.deleted) == 2 and not part.uploads              # both Google-side copies deleted


async def test_same_project_keys_reuse_one_upload(tmp_path):
    big = tmp_path / "v.mp4"; big.write_bytes(b"x" * 100)
    part = MediaPart("file", mime="video/mp4", path=big)
    c, fake, _ = client({"k1": [err(429), err(429)]}, keys=(("same", "k1"), ("same", "k2")))
    orig = fake.__call__
    async def tracking(req):
        if req.url.path.endswith("/upload/v1beta/files"):
            key_of_session[0] = req.headers["x-goog-api-key"]
        return await orig(req)
    c.http = httpx.AsyncClient(transport=httpx.MockTransport(tracking))
    await c.generate([part], system="s")
    assert fake.uploads == ["k1"]                                    # project-wide file: no needless re-upload
