# ShotFinder

Find any screenshot, photo or video by what's written in it or what it shows. Local, offline, multilingual. Windows.

ShotFinder indexes your folders into a single local SQLite file, then searches it with text you remember: a word from a screenshot or a description of a scene. OCR reads text in images, a CLIP image encoder makes every image searchable by meaning, and a multilingual text encoder lets you query in Italian or English against the same visual space. Videos are sampled into frames and indexed the same way. Nothing leaves your machine.

## Status

Phase 5 of 6. Working today:

- Folder scanning, live file watching, hash-based move/rename detection.
- Windows OCR on images, stored in full-text search.
- CLIP visual embeddings for images, stored in a local vector index.
- Video frame indexing: one frame every 5 seconds, stretched to cover the whole file on long videos, near-duplicates dropped, with a timestamp for every hit.
- Local speech transcription with Whisper: videos are searchable by what is said, with the spoken line and its timestamp. Transcribing runs after frames, only while nothing else is pending, and skips silent or very long videos.
- Hybrid search that returns confident text matches and a separate, clearly-labelled "Related by appearance" group, with `--explain` showing the raw text rank, cosine similarity and match source per result.
- A tray app with a Spotlight-style search window, keyboard navigation, preview, copy and reveal. Videos play in-app at the matched moment, two seconds before a spoken line.
- Model download with SHA-256 verification and resume, from the CLI or the tray.

Not built yet: packaging and first-run setup (Phase 6).

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

That fetches about 640 MB into `%LOCALAPPDATA%\ShotFinder\models`: the CLIP image encoder (335 MB), the multilingual text encoder (129 MB), its tokenizer (3 MB) and the Whisper base speech model (145 MB, in `models\whisper-base`). Files are verified against pinned SHA-256 hashes and can resume after an interruption. `python -m shotfinder.cli models status` shows what is present. The CLIP files come from the `models-v1` release and Whisper from `models-v2`.

Without the models, indexing still runs OCR and search still returns text matches. The tray app offers to download them too: when they are missing it shows a notification, and the tray menu has a "Download models" item with progress in the tooltip. Text search works while that runs.

## Run the app

```powershell
python -m shotfinder
```

ShotFinder starts in the tray and indexes in the background; the tooltip shows `Idle`, `Indexing 132/2,048`, `Transcribing lecture.mp4 (34%)`, `Paused` or `Models missing`. `Ctrl+Alt+Space` opens the search window, and so do the tray icon and its menu. A second launch does not start a second copy: it asks the running one to show its window.

If another app already owns `Ctrl+Alt+Space`, ShotFinder tries `Ctrl+Shift+Space`, `Ctrl+Alt+F`, `Ctrl+Shift+F12` and then `Alt+Shift+S`, keeps the first that registers and saves it to `config.json`; the tray tooltip and the first menu line show the active combo. A combo you set yourself is never overwritten: if it is taken, ShotFinder uses a fallback for that session and tells you. The tray menu also pauses or resumes indexing, opens the data folder and quits.

| Key | Action |
| --- | --- |
| Arrows | Move between results |
| `Enter` | Preview the selected file |
| `Ctrl+Enter` | Show the file in Explorer |
| `Ctrl+C` | Copy the image to the clipboard |
| `Esc` | Close the window |

Typing is debounced by 250 ms and searches run off the UI thread, so a slow query never blocks the window. An empty query lists recent files; a query with no hits explains what was searched instead of showing a blank panel.

Results come in two groups. **Matches** are text hits and text-plus-visual hits ordered by reciprocal rank fusion. A filename is text too, so `pulizia disco` finds `Pulizia disco 2026-09-25.mp4` even when nothing inside it matches. A spoken line is text as well: its tile carries a `SPOKEN` badge, shows the transcript with the query highlighted and the line's timestamp, and the preview starts two seconds before it. **Related by appearance** holds visual-only hits above the garbage floor, ordered by cosine and capped at 8; the header says "approximate" because it is. A video tile shows the best-matching frame and a timestamp badge such as `02:14`; the preview starts two seconds before that moment.

In the video preview, `Space` plays or pauses and the left/right arrows jump five seconds. If Windows cannot decode the codec, the window shows the matched still frame and an "Open externally" button instead of a black rectangle. When VLC is the default player, opening externally forwards the start time.

![Empty state](docs/screenshots/empty-state.png)

![Results](docs/screenshots/results.png)

![Preview](docs/screenshots/preview.png)

## Use from the command line

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

The first run watches the folders that exist among the Windows Pictures, Screenshots, Videos and Downloads known folders, so OneDrive redirection and localised folder names are handled. To change them, edit `watched_folders` in `%LOCALAPPDATA%\ShotFinder\config.json` and run `index` again. `--data-dir` points the CLI at a different data directory.

Files that exist only in the cloud are never downloaded or read: they are marked `skipped` with the reason `cloud-only` and picked up on a later scan once they are on disk.

## How video indexing works

Each video is sampled on a grid of `max(5 s, duration / 300)`, measured from the ffmpeg probe before decoding. Short videos keep one frame every five seconds; a two-hour video is sampled every 24 seconds so its end is still covered instead of being cut off at the frame cap. Frames are scaled on the ffmpeg side (bundled through `imageio-ffmpeg`, no system install), so a full-resolution frame is never held in memory. A frame is kept only when its perceptual hash differs from the last kept frame; a static screen recording therefore stores a handful of vectors instead of one per sample, and every video keeps at least its first frame. The 300-frame limit remains as a safety net.

Sampling is resumable. The worker processes at most 30 frames per turn (`video_frames_per_turn`), writes the vectors and the resume cursor in one transaction, then puts the video back in the queue. A long video is indexed in the background without blocking images, and interrupting the app never double-embeds a frame. Videos indexed before this change whose sampling stopped early are detected on the next start and re-sampled; only their frame vectors are reset, so existing speech text survives.

### Speech

After a video's frames are done, and only when no images or frames are waiting, the worker extracts its audio with the bundled ffmpeg to a temporary 16 kHz mono WAV inside the data folder and transcribes it with Whisper `base` (CTranslate2, int8, CPU). The WAV is deleted in `finally`, including after a failure. The model is loaded from the local `models\whisper-base` directory only: `HF_HUB_OFFLINE=1` is set for the process, so a missing file is an error, never a silent download.

Videos without an audio stream, and videos longer than `speech_max_minutes` (default 60), are marked `skipped` with the reason and never queued. Each segment is stored in `text_fts` with `source='speech'` and its start time. Transcription is not resumable per segment: an interrupted run restarts that video from the beginning, and a failed run stays retryable on the next start.

## Known limitations

- **Visual search is approximate.** On the calibration corpus, noise queries reach cosine 0.2503 while real hits fall to 0.2133, so the model cannot separate them with a single cutoff. Visual-only results are therefore shown in a separate, capped "Related by appearance" group rather than as confident matches. See `docs/threshold-calibration.md`. The planned fix is a stronger image/text model (SigLIP 2).
- **Video audio is transcribed only up to 60 minutes per video** (`speech_max_minutes`). Longer videos remain searchable by their frames; raise the limit in `config.json` if needed.
- **Speech accuracy depends on the recording.** Whisper `base` is chosen for speed; strong accents, overlapping speakers and music can produce approximate words. Transcription runs in the detected language and is not translated.
- **In-app video playback depends on Windows codecs.** Common H.264 MP4 and most WebM files play; exotic codecs fall back to the matched still frame and an external player.
- **OCR is Windows-only** (`Windows.Media.Ocr`), and the index covers the watched folders you configure, not the whole disk.
- **The first visual query is slow** while the ONNX runtime and models load; later queries reuse them.

## Privacy

Indexing and search are entirely local. The database, models, logs and thumbnail cache live under `%LOCALAPPDATA%\ShotFinder`; the only network access is the one-time model download. No telemetry, no accounts, no uploads.

## Built for Hack Club Stardance

Built for [Hack Club Stardance](https://stardance.hackclub.com).

## License

MIT. The model files are distributed separately on the `models-v1` and `models-v2` releases; their upstream licenses are listed there. Whisper `base` comes from `Systran/faster-whisper-base` (MIT).
