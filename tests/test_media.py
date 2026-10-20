import subprocess
from pathlib import Path

import pytest

from tgsorter.config import Settings
from tgsorter.media import HAVE_FFMPEG, MediaPart, prepare_attachment, rebalance_inline, probe
from tgsorter.telegram_io import Attachment

pytestmark = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")


def ff(*args):
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", *args], check=True)


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    d = tmp_path_factory.mktemp("m")
    ff("-f", "lavfi", "-i", "testsrc=duration=30:size=640x360:rate=10", "-f", "lavfi", "-i", "sine=frequency=440:duration=30",
       "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(d / "short.mp4"))
    ff("-f", "lavfi", "-i", "testsrc=duration=200:size=640x360:rate=5", "-f", "lavfi", "-i", "sine=frequency=300:duration=200",
       "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(d / "long.mp4"))
    ff("-f", "lavfi", "-i", "testsrc=duration=200:size=320x240:rate=5", "-c:v", "libx264", "-preset", "ultrafast", str(d / "silent.mp4"))
    ff("-f", "lavfi", "-i", "sine=frequency=500:duration=700", "-c:a", "pcm_s16le", str(d / "long.wav"))
    ff("-f", "lavfi", "-i", "testsrc=size=4000x3000:rate=1:duration=1", "-frames:v", "1", str(d / "huge.png"))
    (d / "notes.txt").write_text("Invoice #42 from ACME Ltd, total 1200 EUR")
    (d / "blob.bin").write_bytes(b"\x00\x01" * 100)
    return d


def att(kind, path: Path, mime=None, duration=0, name=None):
    return Attachment(kind, "fid", "uid", path.stat().st_size, mime, name or path.name, duration, "thumbid")


async def test_short_video_is_compressed_inline_low_res(media, tmp_path):
    s = Settings()
    pr = await prepare_attachment(att("video", media / "short.mp4", "video/mp4"), media / "short.mp4", None, tmp_path, s)
    vids = [p for p in pr.parts if p.mime == "video/mp4"]
    assert len(vids) == 1 and vids[0].kind == "inline" and pr.low_res
    info = await probe(vids[0].path)
    assert info.has_audio and info.has_video and 29 < info.duration < 31


async def test_long_video_becomes_audio_plus_keyframes_not_full_video(media, tmp_path):
    s = Settings()
    pr = await prepare_attachment(att("video", media / "long.mp4", "video/mp4"), media / "long.mp4", None, tmp_path, s)
    mimes = [p.mime for p in pr.parts if p.kind != "text"]
    assert mimes.count("audio/mp3") == 1 and mimes.count("image/jpeg") == 5      # 200 s -> 5 frames
    assert "video/mp4" not in mimes and not pr.low_res
    assert any("still frames" in n for n in pr.notes)
    assert all(p.kind == "inline" for p in pr.parts if p.kind != "text")          # nothing needs the Files API
    total = sum(p.size() for p in pr.parts)
    assert total < 1_500_000


async def test_long_video_without_audio_track_gets_frames_only(media, tmp_path):
    pr = await prepare_attachment(att("video", media / "silent.mp4", "video/mp4"), media / "silent.mp4", None, tmp_path, Settings())
    mimes = [p.mime for p in pr.parts if p.kind != "text"]
    assert "audio/mp3" not in mimes and mimes.count("image/jpeg") >= 3


async def test_full_mode_sends_compressed_video_and_truncates(media, tmp_path):
    s = Settings(video_long_mode="full", max_analysis_seconds=60)
    pr = await prepare_attachment(att("video", media / "long.mp4", "video/mp4"), media / "long.mp4", None, tmp_path, s)
    v = [p for p in pr.parts if p.mime == "video/mp4"][0]
    assert 59 < (await probe(v.path)).duration < 61 and pr.low_res
    assert any("only the first 1:00" in n for n in pr.notes)


async def test_long_audio_is_sampled_across_whole_file_with_timestamps(media, tmp_path):
    s = Settings(max_analysis_seconds=600, audio_segments=10)
    a = att("audio", media / "long.wav", "audio/wav", duration=700)
    pr = await prepare_attachment(a, media / "long.wav", None, tmp_path, s)
    clips = [p for p in pr.parts if p.kind == "inline"]
    labels = [p.text for p in pr.parts if p.kind == "text"]
    assert len(clips) == 10 and all(p.mime == "audio/mp3" for p in clips)
    total = 0.0
    for p in clips:
        total += (await probe(p.path)).duration
    assert 595 < total < 606                                   # same total amount as before (600 s)
    assert labels[0].startswith("[audio clip 1/10: 0:05") and "of the original" in labels[-1]
    assert "11:" in labels[-1]                                 # last clip comes from the END of the 11:40 file
    assert sum(p.size() for p in clips) < 2_900_000
    assert any("spread evenly across the whole recording" in n for n in pr.notes)


async def test_short_audio_is_sent_whole(media, tmp_path):
    a = att("audio", media / "long.wav", "audio/wav", duration=700)
    pr = await prepare_attachment(a, media / "long.wav", None, tmp_path, Settings(max_analysis_seconds=800))
    assert [p.kind for p in pr.parts] == ["inline"] and 699 < (await probe(pr.parts[0].path)).duration < 701



async def test_huge_image_is_downscaled(media, tmp_path):
    pr = await prepare_attachment(att("photo", media / "huge.png", "image/png"), media / "huge.png", None, tmp_path, Settings())
    p = pr.parts[0]
    assert p.mime == "image/jpeg" and p.size() < 600_000


async def test_too_big_to_download_degrades_to_thumbnail_and_metadata(media, tmp_path):
    thumb = media / "huge.png"
    a = Attachment("video", "fid", "uid2", 900 * 1048576, "video/mp4", "holiday.mp4", 1800, "thumbid")
    pr = await prepare_attachment(a, None, thumb, tmp_path, Settings())
    assert len([p for p in pr.parts if p.mime == "image/jpeg"]) == 1
    joined = " ".join(pr.notes)
    assert "too large to download" in joined and "holiday.mp4" in joined and "900.0MB" in joined


async def test_text_document_is_read_inline_as_text(media, tmp_path):
    pr = await prepare_attachment(att("document", media / "notes.txt", "text/plain"), media / "notes.txt", None, tmp_path, Settings())
    assert pr.parts[0].kind == "text" and "ACME" in pr.parts[0].text


async def test_pdf_small_inline_big_to_files_api(tmp_path):
    s = Settings(inline_max_bytes=1000)
    small, big = tmp_path / "s.pdf", tmp_path / "b.pdf"
    small.write_bytes(b"%PDF" + b"0" * 100); big.write_bytes(b"%PDF" + b"0" * 5000)
    for f, kind in ((small, "inline"), (big, "file")):
        pr = await prepare_attachment(att("document", f, "application/pdf"), f, None, tmp_path, s)
        assert pr.parts[0].kind == kind


async def test_unknown_binary_is_metadata_only(media, tmp_path):
    pr = await prepare_attachment(att("document", media / "blob.bin", "application/octet-stream"), media / "blob.bin", None, tmp_path, Settings())
    assert not pr.parts and any("filename=blob.bin" in n for n in pr.notes)


def test_rebalance_moves_largest_parts_to_files_api(tmp_path):
    ps = []
    for i, n in enumerate((900, 800, 100)):
        f = tmp_path / f"{i}.bin"; f.write_bytes(b"0" * n)
        ps.append(MediaPart("inline", mime="x/y", path=f))
    rebalance_inline(ps, 1000)
    assert [p.kind for p in ps] == ["file", "inline", "inline"]
