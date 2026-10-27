# ShotFinder

Find any screenshot, photo or video by what's written in it, what it shows, or what's said in it. Local, offline, multilingual. Windows.

ShotFinder indexes your folders into a single local SQLite file, then searches it with text you remember: a word from a screenshot, a description of a scene, or something said in a video. OCR reads text in images, a CLIP image encoder makes every image searchable by meaning, and a multilingual text encoder lets you query in Italian or English against the same visual space. Nothing leaves your machine.

## Status

Phase 2 of 6. Working today, from the command line:

- Folder scanning, live file watching, hash-based move/rename detection.
- Windows OCR on images, stored in full-text search.
- CLIP visual embeddings for images, stored in a local vector index.
- Hybrid search that merges full-text and visual results (reciprocal rank fusion), with `--explain` showing the raw text rank and cosine similarity per result.
- Model download with SHA-256 verification and resume.

Not built yet: the tray app and search window (Phase 3), video frame and speech indexing (Phase 4), packaging and first-run setup (Phases 5-6). Videos are detected and skipped with a note; the UI does not exist, so everything is CLI-only for now.

## Requirements

- Windows 10 or 11, x64.
- Python 3.11 or newer.

## Install

```powershell
git clone https://github.com/brighinatommaso-eng/shotfinder.git
cd shotfinder
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

## Download the models

```powershell
python -m shotfinder.cli models download
```

That fetches about 467 MB into `%LOCALAPPDATA%\ShotFinder\models`: the CLIP image encoder (335 MB), the multilingual text encoder (129 MB) and its tokenizer (3 MB). Files are verified against pinned SHA-256 hashes and can resume after an interruption. `python -m shotfinder.cli models status` shows what is present.

Without the models, indexing still runs OCR and search still returns text matches.

## Use

```powershell
# Scan the watched folders and index new or changed files
python -m shotfinder.cli index

# Search by text in a screenshot, or by what an image shows
python -m shotfinder.cli search "fattura enel"
python -m shotfinder.cli search "spiaggia al tramonto"
python -m shotfinder.cli search "spiaggia al tramonto" --explain

# Most recent indexed files
python -m shotfinder.cli search
```

The first run watches the folders that exist among `Pictures\Screenshots`, `Pictures`, `Videos` and `Downloads`. To change them, edit `watched_folders` in `%LOCALAPPDATA%\ShotFinder\config.json` and run `index` again. `--data-dir` points the CLI at a different data directory.

## Privacy

Indexing and search are entirely local. The database, models and logs live under `%LOCALAPPDATA%\ShotFinder`; the only network access is the one-time model download. No telemetry, no accounts, no uploads.

## Built for Hack Club Stardance

Built for [Hack Club Stardance](https://stardance.hackclub.com).

## License

MIT. The model files are distributed separately on the `models-v1` release; their upstream licenses are listed there.
