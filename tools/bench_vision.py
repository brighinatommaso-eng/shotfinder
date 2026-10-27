"""Measure CLIP image-encoder throughput on real images (Phase 2, Step 1).

Usage::

    python tools/bench_vision.py "C:\\Users\\me\\Pictures\\Screenshots"
    python tools/bench_vision.py <folder> --limit 32 --reps 5 --models-dir models

Reports MEDIAN milliseconds per image over repeated runs. Timings include
preprocessing (decode is outside the loop, resize/crop/normalize is inside),
because that is the per-image cost the indexer pays. Chrome open makes the
numbers noisy; compare medians, not single runs.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shotfinder.config import Config  # noqa: E402
from shotfinder.extractors.vision import (  # noqa: E402
    create_session,
    preprocess_images,
)
from shotfinder.models import IMAGE_MODEL_FILENAME  # noqa: E402

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.replace(",", " ").split() if item.strip()]


def load_images(folder: Path, limit: int) -> list[Image.Image]:
    paths = sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise SystemExit(f"No images found in {folder}")
    paths = paths[:limit]
    images: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as image:
            images.append(ImageOps.exif_transpose(image).convert("RGB"))
    megapixels = sum(image.width * image.height for image in images) / 1e6
    print(f"Loaded {len(images)} image(s) from {folder} ({megapixels:.1f} MP total)")
    return images


def embed_batch(session, images: list[Image.Image], batch_size: int) -> None:
    for start in range(0, len(images), batch_size):
        pixels = preprocess_images(images[start : start + batch_size])
        session.run(["image_embeds"], {"pixel_values": pixels})


def measure(
    images: list[Image.Image],
    model_path: Path,
    *,
    threads: int,
    batch_size: int,
    reps: int,
    allow_spinning: bool = False,
) -> float:
    session = create_session(model_path, threads=threads, allow_spinning=allow_spinning)
    try:
        embed_batch(session, images[:2], batch_size)  # warm up allocations
        samples: list[float] = []
        for _ in range(reps):
            started = time.perf_counter()
            embed_batch(session, images, batch_size)
            elapsed = time.perf_counter() - started
            samples.append(elapsed * 1000.0 / len(images))
        return statistics.median(samples)
    finally:
        del session
        gc.collect()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Folder with real images.")
    parser.add_argument("--limit", type=int, default=32, help="Images to use (default: 32).")
    parser.add_argument("--reps", type=int, default=5, help="Repetitions per config (default: 5).")
    parser.add_argument(
        "--threads", default="1,2,4,6", help="Comma-separated thread counts (default: 1,2,4,6)."
    )
    parser.add_argument(
        "--batches", default="1,8", help="Comma-separated batch sizes (default: 1,8)."
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Folder with the exported ONNX files (default: config models dir).",
    )
    args = parser.parse_args(argv)

    config = Config.load()
    models_dir = args.models_dir or config.models_dir
    model_path = models_dir / IMAGE_MODEL_FILENAME
    if not model_path.is_file():
        raise SystemExit(
            f"{model_path} is missing. Export the models or pass --models-dir / "
            "SHOTFINDER_MODELS_DIR."
        )

    threads = parse_int_list(args.threads)
    batches = parse_int_list(args.batches)
    images = load_images(args.folder, args.limit)

    print(f"Model: {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Config default: onnx_threads={config.onnx_threads}, vision_batch_size={config.vision_batch_size}")
    print(f"Median of {args.reps} run(s), spinning off\n")

    results: dict[tuple[int, int], float] = {}
    for thread_count in threads:
        for batch_size in batches:
            median = measure(
                images,
                model_path,
                threads=thread_count,
                batch_size=batch_size,
                reps=args.reps,
            )
            results[(thread_count, batch_size)] = median
            print(f"  threads={thread_count:2d}  batch={batch_size:2d}  {median:8.2f} ms/image")

    default_batch = config.vision_batch_size
    best = min(results, key=lambda key: results[key])
    print(f"\nFastest: threads={best[0]} batch={best[1]} -> {results[best]:.2f} ms/image")

    default_threads = config.onnx_threads
    print(
        f"\nSpinning comparison at defaults "
        f"(threads={default_threads}, batch={default_batch}):"
    )
    for allow_spinning in (False, True):
        median = measure(
            images,
            model_path,
            threads=default_threads,
            batch_size=default_batch,
            reps=args.reps,
            allow_spinning=allow_spinning,
        )
        state = "on " if allow_spinning else "off"
        print(f"  spinning {state}  {median:8.2f} ms/image")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
