"""Measure CLIP cosine separation for known query/image pairs (dev tool).

Usage::

    python tools/calibrate_threshold.py <folder> \
        --pair "spiaggia al tramonto=tramonto_mare.png" \
        --pair "codice sorgente=editor.png"

For each pair it embeds every image in the folder, prints the cosine of the
expected file and the distribution of all the others, so `min_visual_similarity`
can be chosen from measurements instead of guesses.
"""

from __future__ import annotations

import argparse
import contextlib
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shotfinder.config import Config  # noqa: E402
from shotfinder.extractors.vision import embed_images, embed_text  # noqa: E402

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})


def parse_pairs(values: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for value in values:
        query, separator, filename = value.partition("=")
        if not separator or not query.strip() or not filename.strip():
            raise SystemExit(f'Bad --pair "{value}"; expected "query=filename"')
        pairs.append((query.strip(), filename.strip()))
    return pairs


def embed_folder(
    paths: list[Path], models_dir: Path, threads: int, batch_size: int
) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]
        with contextlib.ExitStack() as stack:
            images = [stack.enter_context(Image.open(path)) for path in chunk]
            vectors.append(
                embed_images(
                    images,
                    models_dir=models_dir,
                    threads=threads,
                    batch_size=batch_size,
                )
            )
    return np.concatenate(vectors, axis=0)


def find_expected(paths: list[Path], filename: str) -> int:
    wanted = filename.replace("\\", "/").lower()
    for index, path in enumerate(paths):
        relative = path.as_posix().lower()
        if path.name.lower() == wanted or relative.endswith("/" + wanted):
            return index
    raise SystemExit(f'Expected image "{filename}" is not in the folder')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Folder with the images to measure.")
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        metavar="QUERY=FILENAME",
        help='Known query and the file it should match, e.g. "spiaggia=tramonto.png".',
    )
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args(argv)

    config = Config.load()
    models_dir = args.models_dir or config.models_dir
    threads = args.threads or config.onnx_threads
    pairs = parse_pairs(args.pair)

    paths = sorted(
        path
        for path in args.folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise SystemExit(f"No images found in {args.folder}")

    print(f"Folder: {args.folder} ({len(paths)} image(s))")
    print(f"Models: {models_dir}, threads: {threads}\n")

    vectors = embed_folder(paths, models_dir, threads, args.batch)
    separations: list[tuple[float, float]] = []

    for query, filename in pairs:
        expected = find_expected(paths, filename)
        vector = embed_text(query, models_dir=models_dir, threads=threads)
        cosines = vectors @ vector
        others = np.delete(cosines, expected)
        expected_cosine = float(cosines[expected])
        rank = int(np.argsort(-cosines).tolist().index(expected)) + 1

        print(f'query="{query}"  expected="{paths[expected].name}"')
        print(f"  expected cosine {expected_cosine:.4f}, rank {rank}/{len(paths)}")
        if others.size:
            print(
                f"  others: min {others.min():.4f}  median {statistics.median(others):.4f}  "
                f"p90 {np.percentile(others, 90):.4f}  max {others.max():.4f}"
            )
            separations.append((expected_cosine, float(others.max())))
        else:
            print("  others: n/a (only one image)")
        print()

    if separations:
        lowest_expected = min(expected for expected, _ in separations)
        highest_other = max(other for _, other in separations)
        print(
            f"Separation: lowest expected {lowest_expected:.4f}, "
            f"highest competing {highest_other:.4f}"
        )
        if highest_other < lowest_expected:
            print(
                f"Set min_visual_similarity above {highest_other:.4f} and below "
                f"{lowest_expected:.4f}."
            )
        else:
            print(
                "The pairs overlap; a single absolute threshold cannot separate "
                "them. Check the images or rely on the relative margin."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
