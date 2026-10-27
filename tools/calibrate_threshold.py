"""Measure CLIP cosine separation for known query/image pairs (dev tool).

Usage::

    python tools/calibrate_threshold.py "C:\\Users\\me\\Pictures" --recursive \
        --pair "spiaggia al tramonto=tramonto_mare.png" \
        --pair "codice sorgente=editor.png"

For each pair it embeds every indexed-looking image below the folders, prints
the cosine of the expected file and the distribution of all the others, so
``min_visual_similarity`` can be chosen from measurements instead of guesses.
The z-score column shows whether the hit stands out from the corpus at all;
when the max competing cosine exceeds the lowest expected cosine, an absolute
threshold cannot separate signal from noise and the z-score is the fallback.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shotfinder.config import Config  # noqa: E402
from shotfinder.exclusions import (  # noqa: E402
    Exclusions,
    is_cloud_only,
    is_small_image,
)
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


def collect_images(
    folders: list[Path], config: Config, *, recursive: bool
) -> list[Path]:
    """Images the indexer would embed: supported, not excluded, not cloud-only."""
    roots = [folder for folder in folders if folder.is_dir()]
    exclusions = Exclusions(config, roots)
    seen: set[str] = set()
    paths: list[Path] = []
    for folder in roots:
        candidates: list[Path] = []
        if recursive:
            for dirpath, dirnames, filenames in os.walk(folder):
                current = Path(dirpath)
                dirnames[:] = [
                    name
                    for name in dirnames
                    if not exclusions.excludes_dir(current / name)
                ]
                candidates.extend(current / name for name in filenames)
        else:
            candidates = [path for path in folder.iterdir() if path.is_file()]
        for path in candidates:
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            key = os.path.normcase(os.path.abspath(os.fspath(path)))
            if key in seen or is_cloud_only(path) or is_small_image(path):
                continue
            seen.add(key)
            paths.append(path)
    return sorted(paths)


def find_expected(paths: list[Path], filename: str) -> int:
    """Match a relative path, or a basename that occurs exactly once."""
    wanted = filename.replace("\\", "/").lower().strip("/")
    relative = [
        index
        for index, path in enumerate(paths)
        if path.as_posix().lower().endswith("/" + wanted)
    ]
    if len(relative) == 1:
        return relative[0]
    by_name = [index for index, path in enumerate(paths) if path.name.lower() == wanted]
    if len(by_name) == 1:
        return by_name[0]
    matches = relative or by_name
    if matches:
        options = "\n  ".join(str(paths[index]) for index in matches)
        raise SystemExit(
            f'Expected image "{filename}" is ambiguous; use a relative path:\n  {options}'
        )
    raise SystemExit(f'Expected image "{filename}" is not in the corpus')


def embed_corpus(
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "folders", type=Path, nargs="+", help="Folders with the images to measure."
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Walk subfolders, applying the indexer's directory exclusions.",
    )
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
    parser.add_argument(
        "--noise",
        action="append",
        default=[],
        metavar="QUERY",
        help="A query with no correct file; prints the strongest false positive.",
    )
    args = parser.parse_args(argv)

    config = Config.load()
    models_dir = args.models_dir or config.models_dir
    threads = args.threads or config.onnx_threads
    pairs = parse_pairs(args.pair)

    paths = collect_images(args.folders, config, recursive=args.recursive)
    if not paths:
        raise SystemExit("No images found in " + ", ".join(map(str, args.folders)))

    print(
        f"Corpus: {len(paths)} image(s) from {len(args.folders)} folder(s)"
        + (" (recursive)" if args.recursive else "")
    )
    print(f"Models: {models_dir}, threads: {threads}\n")

    vectors = embed_corpus(paths, models_dir, threads, args.batch)
    rows: list[tuple[str, Path, float, int, float, float, float, float]] = []
    separations: list[tuple[float, float]] = []

    for query, filename in pairs:
        expected = find_expected(paths, filename)
        vector = embed_text(query, models_dir=models_dir, threads=threads)
        cosines = vectors @ vector
        others = np.delete(cosines, expected)
        expected_cosine = float(cosines[expected])
        rank = int(np.argsort(-cosines).tolist().index(expected)) + 1
        if others.size:
            mean = float(others.mean())
            std = float(others.std())
            z_score = (expected_cosine - mean) / std if std > 1e-9 else float("inf")
            rows.append(
                (
                    query,
                    paths[expected],
                    expected_cosine,
                    rank,
                    float(others.max()),
                    float(np.percentile(others, 90)),
                    z_score,
                    float(statistics.median(others)),
                )
            )
            separations.append((expected_cosine, float(others.max())))
        else:
            rows.append(
                (query, paths[expected], expected_cosine, rank, 0.0, 0.0, 0.0, 0.0)
            )

    width = max(len(row[0]) for row in rows)
    print(
        f'{"query".ljust(width)}  {"expected cosine":>15}  {"rank":>7}  '
        f'{"others max":>10}  {"p90":>7}  {"z":>6}'
    )
    print("-" * (width + 58))
    for query, path, cosine, rank, other_max, p90, z_score, _median in rows:
        print(
            f"{query.ljust(width)}  {cosine:>15.4f}  {f'{rank}/{len(paths)}':>7}  "
            f"{other_max:>10.4f}  {p90:>7.4f}  {z_score:>6.2f}   {path.name}"
        )
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
                f"The expected file always outranks every other image: set "
                f"min_visual_similarity above {highest_other:.4f} and below "
                f"{lowest_expected:.4f}."
            )
        else:
            print(
                "Some expected files are outranked by related images (same scene "
                "or subject), so this metric cannot set an exact-file threshold. "
                "Judge the noise queries below instead."
            )

    noise_rows: list[tuple[str, float, float, Path]] = []
    if args.noise:
        print("\nNoise queries (no correct file):")
        print(f'{"query".ljust(width)}  {"top cosine":>10}  {"z":>6}  top file')
        print("-" * (width + 58))
        for query in args.noise:
            vector = embed_text(query, models_dir=models_dir, threads=threads)
            cosines = vectors @ vector
            best = int(np.argmax(cosines))
            others = np.delete(cosines, best)
            mean = float(others.mean())
            std = float(others.std())
            z_score = (float(cosines[best]) - mean) / std if std > 1e-9 else float("inf")
            noise_rows.append((query, float(cosines[best]), z_score, paths[best]))
            print(
                f"{query.ljust(width)}  {float(cosines[best]):>10.4f}  "
                f"{z_score:>6.2f}   {paths[best].name}"
            )
        print()

    if separations and noise_rows:
        lowest_expected = min(expected for expected, _ in separations)
        lowest_expected_z = min(row[6] for row in rows)
        noise_cosine = max(row[1] for row in noise_rows)
        noise_z = max(row[2] for row in noise_rows)
        print(
            f"Noise ceiling: cosine {noise_cosine:.4f}, z {noise_z:.2f}; "
            f"weakest expected hit: cosine {lowest_expected:.4f}, z {lowest_expected_z:.2f}"
        )
        if noise_cosine < lowest_expected:
            print(
                f"An absolute floor above {noise_cosine:.4f} rejects every measured "
                f"noise hit while keeping all expected files."
            )
        else:
            print(
                "Noise overlaps the weakest expected hit, and the z-score ranges "
                "overlap too, so no single absolute or z gate separates them. "
                "Keep the absolute floor as a noise ceiling and let text search "
                "cover the weak visual cases."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
