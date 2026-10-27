"""Video sampling: ffmpeg decoding, scene dedup, cap and resumable cursor."""

from __future__ import annotations

from pathlib import Path

import pytest

from shotfinder.extractors import video
from shotfinder.extractors.video import (
    FRAME_INTERVAL,
    FrameCursor,
    VideoUnavailableError,
    extract_frame,
    hamming_distance,
    hash_from_hex,
    hash_to_hex,
    iter_frames,
    perceptual_hash,
)


def test_scene_cut_keeps_far_fewer_frames_than_sampled(scene_video: Path) -> None:
    cursor = FrameCursor()

    frames = list(iter_frames(scene_video, cursor, max_side=64))

    sampled = int(80 / FRAME_INTERVAL)
    assert len(frames) >= 2
    assert len(frames) < sampled / 2
    assert frames[0].timestamp == 0.0
    assert all(frame.image.size == (64, 64) for frame in frames)
    assert cursor.exhausted is True
    assert cursor.kept == len(frames)


def test_a_static_stretch_is_not_kept_twice(scene_video: Path) -> None:
    frames = list(iter_frames(scene_video, FrameCursor(), max_side=64))

    timestamps = [frame.timestamp for frame in frames]
    # The cut is at 20 s; every blue frame after it hashes the same.
    assert any(abs(timestamp - 20.0) < 0.5 for timestamp in timestamps)
    assert len([timestamp for timestamp in timestamps if timestamp >= 20.0]) == 1


def test_resume_continues_after_a_turn(scene_video: Path) -> None:
    cursor = FrameCursor()

    first = list(iter_frames(scene_video, cursor, max_side=64, max_frames=1))
    assert [frame.timestamp for frame in first] == [0.0]
    assert cursor.next_ts == FRAME_INTERVAL
    assert cursor.exhausted is False

    second = list(iter_frames(scene_video, cursor, max_side=64, max_frames=1))
    assert [frame.timestamp for frame in second] == [20.0]
    assert cursor.kept == 2
    assert cursor.exhausted is False

    third = list(iter_frames(scene_video, cursor, max_side=64, max_frames=1))
    assert third == []
    assert cursor.exhausted is True
    assert cursor.kept == 2


def test_kept_frame_cap_stops_sampling(
    scene_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(video, "MAX_KEPT_FRAMES", 2)
    cursor = FrameCursor()

    frames = list(iter_frames(scene_video, cursor, max_side=64))

    assert len(frames) == 2
    assert cursor.kept == 2
    assert cursor.exhausted is False


def test_extract_frame_picks_the_requested_scene(scene_video: Path) -> None:
    first = extract_frame(scene_video, 5, max_side=64)
    blue = extract_frame(scene_video, 30, max_side=64)

    assert first is not None and blue is not None
    red, green, blue_channel = blue.getpixel((32, 24))
    assert blue_channel > 200
    assert blue_channel > red + 100
    assert first.getpixel((32, 24)) != (0, 0, 250)


def test_unreadable_video_raises(tmp_path: Path) -> None:
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video at all")

    with pytest.raises(VideoUnavailableError):
        list(iter_frames(broken, FrameCursor(), max_side=64))


def test_perceptual_hash_roundtrip() -> None:
    assert hash_from_hex(hash_to_hex(123456789)) == 123456789
    assert hash_from_hex(None) is None
    assert hash_from_hex("not-hex") is None
    assert hamming_distance(0b1010, 0b0011) == 2


def test_identical_images_hash_the_same() -> None:
    from PIL import Image

    image = Image.new("RGB", (32, 32), (10, 120, 200))

    assert perceptual_hash(image) == perceptual_hash(image.copy())
