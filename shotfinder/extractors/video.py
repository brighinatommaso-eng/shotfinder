"""Video frame sampling and deduplication (Phase 4).

ffmpeg comes from ``imageio-ffmpeg``, so no system install is required. Frames
are decoded straight to a fixed 448x448 RGB stream on the ffmpeg side (scaled,
then letterboxed), so a full-resolution frame is never materialised in Python.

Sampling is resumable: :class:`FrameCursor` carries the timestamp to resume
from, how many frames were already kept and the hash of the last kept frame.
The cursor is mutated while frames are decoded, so a long video can be
processed in several short turns without losing dedup state.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterator

from PIL import Image

log = logging.getLogger(__name__)

FRAME_INTERVAL = 5.0
FRAME_MAX_SIDE = 448
THUMBNAIL_MAX_SIDE = 256
HASH_DISTANCE = 6
MAX_KEPT_FRAMES = 300
_PIXEL_FORMAT = "rgb24"
_DURATION_RE = re.compile(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)")


class VideoUnavailableError(RuntimeError):
    """Raised when ffmpeg is missing or cannot decode a file."""


@dataclass(frozen=True, slots=True)
class MediaInfo:
    """What ffmpeg reports before any frame is decoded."""

    duration: float | None
    has_audio: bool


def probe_media(path: str | Path) -> MediaInfo:
    """Read duration and audio presence from the bundled ffmpeg.

    ``imageio-ffmpeg`` ships ffmpeg but not ffprobe; ``ffmpeg -i`` prints the
    same stream header to stderr and exits without decoding.
    """
    executable = ffmpeg_path()
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        result = subprocess.run(
            [executable, "-hide_banner", "-i", str(path)],
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=60,
            creationflags=creationflags,
        )
    except (OSError, subprocess.TimeoutExpired):
        log.debug("Could not probe %s", path, exc_info=True)
        return MediaInfo(None, False)
    text = result.stderr.decode("utf-8", "replace")
    match = _DURATION_RE.search(text)
    duration = None
    if match:
        duration = (
            int(match.group(1)) * 3600
            + int(match.group(2)) * 60
            + float(match.group(3))
        )
    has_audio = any(
        "Stream #" in line and "Audio:" in line for line in text.splitlines()
    )
    return MediaInfo(duration, has_audio)


def sampling_interval(duration: float | None) -> float:
    """Spread at most :data:`MAX_KEPT_FRAMES` samples over the whole video.

    Short videos keep the 5 s grid; a long one gets a coarser grid so its end is
    still covered instead of being cut off at the frame cap.
    """
    if duration is None or duration <= 0:
        return FRAME_INTERVAL
    return max(FRAME_INTERVAL, duration / MAX_KEPT_FRAMES)


@dataclass(slots=True)
class FrameCursor:
    """Durable position in a video sampling run."""

    next_ts: float = 0.0
    kept: int = 0
    last_hash: str | None = None
    # True when the last call reached the end of the file (or failed) instead
    # of stopping at its per-turn frame budget.
    exhausted: bool = False


@dataclass(frozen=True, slots=True)
class Frame:
    """One kept frame: its position in the video, pixels and perceptual hash."""

    timestamp: float
    image: Image.Image
    dhash: int


_FFMPEG_LOCK = threading.Lock()
_FFMPEG: str | None = None


def ffmpeg_path() -> str:
    """Locate the bundled ffmpeg once per process."""
    global _FFMPEG
    with _FFMPEG_LOCK:
        if _FFMPEG is None:
            try:
                import imageio_ffmpeg
            except ImportError as exc:  # pragma: no cover - dependency is pinned
                raise VideoUnavailableError(
                    "imageio-ffmpeg is not installed; video frames need it"
                ) from exc
            _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
        return _FFMPEG


def perceptual_hash(image: Image.Image) -> int:
    """64-bit difference hash: robust to scaling, sensitive to scene changes."""
    small = image.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    pixels = small.tobytes()
    bits = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            bits = (bits << 1) | (
                1 if pixels[offset + column] > pixels[offset + column + 1] else 0
            )
    return bits


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def hash_to_hex(digest: int) -> str:
    return f"{digest:016x}"


def hash_from_hex(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value, 16)
    except ValueError:
        return None


def _spawn(args: list[str]) -> subprocess.Popen[bytes]:
    executable = ffmpeg_path()
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        return subprocess.Popen(
            [executable, "-hide_banner", "-loglevel", "error", *args],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
        )
    except OSError as exc:
        raise VideoUnavailableError(f"could not start ffmpeg: {exc}") from exc


def _stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.kill()
    for stream in (process.stdout, process.stderr):
        if stream is not None:
            try:
                stream.close()
            except OSError:  # pragma: no cover - closing a dead pipe
                pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:  # pragma: no cover - kill already sent
        log.warning("ffmpeg did not exit after being killed")


def _collect_stderr(
    process: subprocess.Popen[bytes],
) -> tuple[list[bytes], threading.Thread]:
    """Drain stderr on a thread so a noisy file cannot fill the pipe."""
    chunks: list[bytes] = []

    def read() -> None:
        stream = process.stderr
        if stream is None:
            return
        try:
            chunks.append(stream.read() or b"")
        except OSError:  # pragma: no cover - closed while reading
            pass

    thread = threading.Thread(target=read, daemon=True)
    thread.start()
    return chunks, thread


def _decode_error(chunks: list[bytes]) -> str:
    payload = b"".join(chunks)
    lines = payload.decode("utf-8", "replace").strip().splitlines()
    return lines[0] if lines else ""


def iter_frames(
    path: str | Path,
    cursor: FrameCursor,
    *,
    interval: float = FRAME_INTERVAL,
    max_side: int = FRAME_MAX_SIDE,
    max_frames: int | None = None,
) -> Iterator[Frame]:
    """Yield kept frames from ``cursor.next_ts``, at most ``max_frames``.

    A frame is kept when its perceptual hash differs from the last kept frame by
    at least :data:`HASH_DISTANCE` bits; the first frame is always kept. The
    cursor is advanced after every decoded frame, so a later call resumes at the
    next sample even when the previous turn ended inside a static scene.
    """
    if max_frames is not None and max_frames <= 0:
        cursor.exhausted = False
        return
    if cursor.kept >= MAX_KEPT_FRAMES:
        cursor.exhausted = False
        return

    path = Path(path)
    size = max(16, int(max_side))
    start = max(0.0, float(cursor.next_ts))
    cursor.exhausted = False
    scale = (
        f"fps=1/{interval:g},"
        f"scale={size}:{size}:force_original_aspect_ratio=decrease,"
        f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2:color=black"
    )
    process = _spawn(
        [
            "-nostdin",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(path),
            "-an",
            "-sn",
            "-vf",
            scale,
            "-f",
            "rawvideo",
            "-pix_fmt",
            _PIXEL_FORMAT,
            "pipe:1",
        ]
    )
    frame_bytes = size * size * 3
    stderr, stderr_thread = _collect_stderr(process)
    previous = hash_from_hex(cursor.last_hash)
    kept = 0
    sampled = 0
    read_any = False
    natural_end = False
    try:
        while True:
            if max_frames is not None and kept >= max_frames:
                break
            if cursor.kept >= MAX_KEPT_FRAMES:
                break
            buffer = process.stdout.read(frame_bytes) if process.stdout else b""
            if not buffer or len(buffer) < frame_bytes:
                natural_end = True
                cursor.exhausted = True
                break
            read_any = True
            timestamp = start + sampled * interval
            sampled += 1
            cursor.next_ts = timestamp + interval
            image = Image.frombytes("RGB", (size, size), buffer)
            digest = perceptual_hash(image)
            if (
                previous is not None
                and hamming_distance(previous, digest) < HASH_DISTANCE
            ):
                continue
            previous = digest
            cursor.kept += 1
            cursor.last_hash = hash_to_hex(digest)
            kept += 1
            yield Frame(timestamp, image, digest)
    finally:
        if natural_end:
            stderr_thread.join(timeout=5)
        _stop(process)

    if natural_end:
        error = _decode_error(stderr)
        if error:
            if not read_any:
                raise VideoUnavailableError(
                    f"ffmpeg could not read {path.name}: {error}"
                )
            log.warning("Stopped decoding %s early: %s", path.name, error)


def extract_frame(
    path: str | Path,
    timestamp: float,
    *,
    max_side: int = THUMBNAIL_MAX_SIDE,
) -> Image.Image | None:
    """Decode a single frame for a thumbnail or a preview fallback."""
    size = max(16, int(max_side))
    process = _spawn(
        [
            "-nostdin",
            "-ss",
            f"{max(0.0, float(timestamp)):.3f}",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-an",
            "-sn",
            "-vf",
            f"scale={size}:{size}:force_original_aspect_ratio=decrease",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "pipe:1",
        ]
    )
    stderr, _thread = _collect_stderr(process)
    try:
        payload = process.stdout.read() if process.stdout else b""
    finally:
        _stop(process)
    if not payload:
        error = _decode_error(stderr)
        if error:
            log.debug("Could not decode %s at %.1fs: %s", path, timestamp, error)
        return None
    try:
        with Image.open(BytesIO(payload)) as image:
            return image.convert("RGB")
    except Exception:
        log.debug("Could not decode a frame from %s at %.1fs", path, timestamp)
        return None
