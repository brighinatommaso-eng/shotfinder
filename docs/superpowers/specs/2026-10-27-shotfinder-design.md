# ShotFinder — Design Spec

**Date:** 2026-10-27
**Context:** Hack Club Stardance — "Frictionless" quest (build a QoL tool)
**Target platform:** Windows 10/11 (x64), CPU-only
**Time budget:** 5–10 hours
**Status:** Approved design, ready for implementation

---

## 1. Problem

People accumulate thousands of screenshots, photos and videos with useless names
(`Screenshot 2026-10-12 143322.png`, `IMG_4821.jpg`, `VID_20260801.mp4`).
Finding "that screenshot with the train time", "the photo of the sunset at the beach"
or "the video where someone says *meeting on Thursday*" means scrolling for minutes.

## 2. Solution

A local, offline Windows tray app that indexes the user's media folders in the
background and lets them find any file by typing what is **written in it**, what
it **shows**, or what is **said in it** — in any language — from a global hotkey.

## 3. Goals / Non-goals

**Goals**
- Search images by visible text (OCR) and by visual content (semantic, multilingual).
- Search videos by visual content (sampled frames) and by spoken words (transcription).
- Video results open at the exact matching timestamp.
- 100% local and offline after first model download. No telemetry, no cloud.
- Lightweight: CPU-only, quantized models, background indexing at low priority.

**Non-goals (YAGNI)**
- No macOS/Linux support in v1.
- No GPU path, no face recognition, no tagging/albums, no cloud sync.
- No file renaming/moving (read-only on user files — never modify them).
- No audio-only files (mp3, wav) in v1.

## 4. Tech stack

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Fastest to build, all libs available |
| Text in images | Windows built-in OCR (`Windows.Media.Ocr`) via `winrt-*` packages | Zero download, fast, good quality |
| Visual search | CLIP ViT-B/32 image encoder, ONNX int8, `onnxruntime` | Small, CPU-friendly |
| Multilingual query | `sentence-transformers/clip-ViT-B-32-multilingual-v1` text encoder, exported to ONNX int8 | Aligned with ViT-B/32 image space; Italian queries work |
| Speech | `faster-whisper`, model `base`, `compute_type="int8"` | Best speed/quality on CPU |
| Video/audio decode | `ffmpeg` via `imageio-ffmpeg` (bundled binary) | No system install needed |
| Storage | SQLite single file: FTS5 for text, `sqlite-vec` for vectors | One file, no server |
| File watching | `watchdog` | Standard |
| UI | `PySide6` (tray + search window + preview player) | Native-feeling, has media player |
| Global hotkey | `pynput` `GlobalHotKeys` | Works without admin |
| Packaging | PyInstaller (one-folder) | Models NOT bundled, downloaded on first run |

**Hard constraint:** the multilingual text encoder is only aligned with the
original CLIP ViT-B/32 image encoder. Do not swap the image model (e.g. MobileCLIP)
without swapping the text model too.

## 5. Architecture

```
Watcher (watchdog) ──► Index queue (SQLite, per-file state)
                              │  low-priority worker thread
             ┌────────────────┼─────────────────────┐
             ▼                ▼                     ▼
          Image            Video (visual)       Video (speech)
     Win OCR + CLIP    ffmpeg 1 frame / 5 s   ffmpeg extract audio
                       dedup near-identical   faster-whisper base
                       frames → CLIP          segments + timestamps
             └────────────────┬─────────────────────┘
                              ▼
                    index.db (single SQLite file)
                              ▲
        Search window (global hotkey) ── hybrid query ── results grid
```

### 5.1 Project layout

```
shotfinder/
  __main__.py          # entry point: starts tray app
  config.py            # paths, watched folders, hotkey, tunables (dataclass + JSON file)
  models.py            # model registry, first-run download with progress, paths
  index.py             # ONLY module that touches SQLite (schema, CRUD, queries)
  watcher.py           # watchdog observer + initial full scan → enqueue
  worker.py            # pulls from queue, dispatches to extractors, writes via index.py
  search.py            # hybrid search + ranking
  extractors/
    ocr.py             # image path -> list[str] text lines (Windows OCR)
    vision.py          # image (PIL) -> np.ndarray[512] normalized; text -> np.ndarray[512]
    video.py           # video path -> iterator of (timestamp_s, PIL image), deduped
    speech.py          # video path -> list[(start_s, end_s, text)]
  ui/
    tray.py            # tray icon, menu (pause/resume indexing, settings, quit), status
    search_window.py   # frameless popup: input + results grid with thumbnails
    preview.py         # image viewer + QMediaPlayer seeking to timestamp
    settings.py        # pick watched folders, toggle video/speech indexing
tools/
  export_models.py     # one-off: export + quantize ONNX models (dev only)
tests/
```

Each extractor is a **pure function of a file → data**. Extractors never know about
the database; `worker.py` glues them to `index.py`.

### 5.2 Data model (`index.db`)

```sql
CREATE TABLE files (
  id INTEGER PRIMARY KEY,
  path TEXT UNIQUE NOT NULL,
  hash TEXT NOT NULL,            -- xxhash of first 1MB + file size (survives move/rename)
  kind TEXT NOT NULL,            -- 'image' | 'video'
  size INTEGER, mtime REAL, created REAL,
  status TEXT NOT NULL,          -- 'pending' | 'done' | 'error' | 'skipped'
  error TEXT,
  stages_done TEXT DEFAULT ''    -- e.g. 'ocr,vision' | 'frames,speech'
);

-- text from OCR and speech; ts = NULL for images, seconds for video
CREATE VIRTUAL TABLE text_fts USING fts5(
  content, file_id UNINDEXED, ts UNINDEXED, source UNINDEXED,   -- source: 'ocr' | 'speech'
  tokenize = 'unicode61 remove_diacritics 2'
);

-- one vector per image, one per kept video frame
CREATE TABLE vec_meta (rowid INTEGER PRIMARY KEY, file_id INTEGER, ts REAL);
CREATE VIRTUAL TABLE vec_items USING vec0(embedding float[512]);
```

- **Hash-based identity:** if a file is moved/renamed and the hash matches an
  existing row, update `path` instead of re-indexing.
- **Deleted files:** watcher removes rows (cascade to fts/vec by file_id).
- `stages_done` makes indexing resumable per stage (e.g. frames done, speech pending).

### 5.3 Indexing pipeline

- Supported: images `png jpg jpeg webp bmp heic*`, videos `mp4 mov mkv webm avi`.
  (*heic only if `pillow-heif` loads; otherwise skipped.)
- Order of work: images first (fast, high value), then video frames, then speech last.
- **Images:** OCR → insert lines into `text_fts`; CLIP embed → `vec_items`.
  Skip OCR result if empty. Downscale to max 1600px before OCR, 224px for CLIP.
- **Video frames:** sample 1 frame every 5 s with ffmpeg; compute a cheap
  perceptual hash (e.g. `imagehash.phash` on 64px) and drop frames whose hash
  distance to the last kept frame is < 6. Embed kept frames. Cap at 300 kept frames
  per video.
- **Speech:** extract 16 kHz mono WAV with ffmpeg to a temp file, run
  faster-whisper `base` int8 with VAD filter on, language auto-detect. Store each
  segment as an FTS row with `ts = start`. Skip videos with no audio stream.
  Configurable max duration (default 60 min) — longer videos get frames only.
- **Politeness:** worker thread lowered to `BELOW_NORMAL` priority (via `psutil`);
  pauses while system CPU > 70% for 10 s; `onnxruntime` and whisper limited to
  `max(1, cpu_count // 2)` threads. Tray menu can pause/resume.

### 5.4 Search

1. Query text → **FTS5** match (prefix-friendly: each token gets `*`), BM25 score.
2. Query text → multilingual text embedding → **sqlite-vec** KNN (k=50), cosine.
3. **Merge with Reciprocal Rank Fusion** (k=60) across both lists, grouped by
   `file_id`; for videos keep the best-matching `ts` as the jump point.
4. Return top 40: `{file_id, path, kind, ts, snippet, score, match_type}`.
5. Empty query → show 40 most recent files.

Queries run on a separate thread; UI debounces input by 250 ms.

### 5.5 UI

- **Tray icon** with status tooltip ("Indexing 132/2,048", "Idle", "Paused") and
  menu: Open search, Pause/Resume, Settings, Quit.
- **Global hotkey** (default `Ctrl+Alt+Space`, configurable) toggles a frameless,
  centered search window (Spotlight-like): input on top, grid of thumbnails below.
- Each result shows: thumbnail, filename, a badge (`TEXT` / `VISUAL` / `SPOKEN`),
  and for videos the timestamp (`02:14`) and transcript snippet if any.
- `Enter` / double-click → preview pane: images shown full; videos played in-app
  with `QMediaPlayer`, seeked to `ts - 2 s`. `Ctrl+Enter` → open in Explorer
  (`explorer /select,`). `Esc` closes.
- Thumbnails cached in `%LOCALAPPDATA%\ShotFinder\thumbs\<hash>.jpg`.

### 5.6 First run

1. Settings window: preselect `Pictures\Screenshots`, `Pictures`, `Videos`,
   `Downloads`; user can add/remove.
2. Model download with progress bar into `%LOCALAPPDATA%\ShotFinder\models`
   (~CLIP image + multilingual text + whisper base ≈ 350–400 MB). Indexing images
   with OCR can start before vision/speech models finish downloading.
3. Initial full scan, then live watching.

## 6. Error handling

- Any extractor failure → mark that file `status='error'` with message, continue.
  Never crash the worker on a single bad file.
- Corrupt/unreadable media, zero-byte files, files locked by another process:
  retry once after 30 s, then `error`.
- Windows OCR language pack missing → fall back to `en-US` recognizer; if no
  recognizer available at all, disable OCR stage and show a tray notification.
- Model download failure → retry with backoff; the app still works with the
  stages whose models are present.
- Paths: always use `pathlib`, support long paths and non-ASCII names.
- Logs: rotating file in `%LOCALAPPDATA%\ShotFinder\logs\`.

## 7. Testing

- `tests/fixtures/`: a screenshot with known text, a photo (beach/sunset), a
  10-second video with a known spoken sentence and a scene change.
- Unit tests per extractor (OCR finds the known word; CLIP ranks "beach" photo
  above the screenshot for query `spiaggia`; video dedup keeps ≥2 frames;
  whisper transcript contains the known word).
- `index.py` tests on a temp DB: insert, move-by-hash, delete cascade, FTS + vec query.
- `search.py` test: RRF merge ordering on synthetic inputs.
- Manual smoke test checklist in README.
- Heavy tests (models) marked `@pytest.mark.slow`.

## 8. Delivery plan

| Phase | Scope | Est. |
|---|---|---|
| 0 | Scaffold, config, `index.py` schema, watcher + queue + worker skeleton, CLI `python -m shotfinder.cli search "..."` | 1 h |
| 1 | Windows OCR for images → FTS search working end-to-end in CLI | 1.5 h |
| 2 | Model export/download, CLIP image + multilingual text, vec search, RRF hybrid | 2 h |
| 3 | PySide6 tray, hotkey, search window, thumbnails, image preview | 2 h |
| 4 | Video frames (ffmpeg, dedup, CLIP), in-app player seeking | 1 h |
| 5 | Speech (faster-whisper) + transcript snippets in results | 1–1.5 h |
| 6 | Settings, first-run flow, PyInstaller build, README + demo GIF | 1 h |

Total ≈ 9.5–10 h — at the top of the budget. **Cut order if late:** Phase 5 →
Phase 4 → settings UI (hardcode folders). Phases 0–3 alone are a shippable product.

**Stardance requirements:** keep Hackatime running in the editor, write a devlog
per session, public GitHub repo, release `.zip` on GitHub Releases, README with
demo GIF.

## 9. Open risks

- Exporting the multilingual text encoder to ONNX (DistilBERT + dense projection)
  is the most fragile step. Phase 2 starts with a 20-minute spike; fallback is
  loading it via `sentence-transformers` (heavier: pulls in torch CPU).
- `sqlite-vec` requires `enable_load_extension`; verify the Python build supports
  it on day one (python.org Windows builds do).
- Whisper `base` on weak CPUs may run near real-time; the duration cap and
  idle-only processing keep it from hurting UX.
