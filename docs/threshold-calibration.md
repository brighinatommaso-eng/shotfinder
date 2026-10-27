# Visual threshold calibration (2026-10-27)

Corpus: 56 indexed images from the real watched folders (OneDrive Pictures,
Screenshots, Videos, Downloads), recursive, excluding caches/projects/tiny
images. Models: `clip_image_fp32` + `clip_text_int8`, 4 threads. Queries were
picked from OCR text and filenames in the index.

Command (paths shortened):

```powershell
python tools/calibrate_threshold.py <pictures> <screenshots> <videos> <downloads> `
  --recursive --models-dir models `
  --pair "spider robot=...Review (BQ).jpg" `
  --pair "installazione debian=WIN_20260810_20_53_47_Pro.jpg" `
  ... `
  --noise spiaggia --noise automobile --noise gatto --noise montagna --noise neve
```

## Expected-file results

| query | expected cosine | rank | others max | p90 | z |
|---|---:|---:|---:|---:|---:|
| spider robot | 0.2677 | 4/56 | 0.2795 | 0.2289 | 2.37 |
| hexapod assembly | 0.2507 | 2/56 | 0.3055 | 0.2442 | 1.45 |
| claude code handbook | 0.3781 | 1/56 | 0.3625 | 0.2503 | 5.26 |
| github release assets | 0.2762 | 1/56 | 0.2687 | 0.2389 | 3.30 |
| git project folders | 0.2646 | 1/56 | 0.2574 | 0.2486 | 1.89 |
| installazione debian | 0.2264 | 31/56 | 0.2430 | 0.2388 | 0.21 |
| passphrase rete wifi | 0.2610 | 3/56 | 0.2658 | 0.2564 | 1.48 |
| proxy http | 0.2600 | 1/56 | 0.2543 | 0.2506 | 1.71 |
| partizioni disco | 0.2195 | 32/56 | 0.2597 | 0.2374 | 0.07 |
| climbing festival | 0.2133 | 13/56 | 0.2737 | 0.2358 | 0.63 |
| riyadh landscape | 0.2741 | 1/56 | 0.2636 | 0.2329 | 2.93 |
| dominio internet | 0.2590 | 2/56 | 0.2599 | 0.2480 | 1.50 |

`z` is the expected cosine's z-score against the other images' cosines for the
same query.

## Noise queries (no correct file)

| query | top cosine | z | top file |
|---|---:|---:|---|
| spiaggia | 0.2383 | 1.92 | DSCN0160.JPG |
| automobile | 0.2503 | 2.38 | Pezzo 2 Ragno Robot.jpeg |
| gatto | 0.2355 | 1.94 | Stemma Casalotto.png |
| montagna | 0.2478 | 1.73 | WIN_20260810_21_46_14_Pro.jpg |
| neve | 0.2403 | 1.43 | Stemma Canali.png |

## Conclusion

Noise tops out at cosine **0.2503** (z 2.38); the weakest real hit is **0.2133**
(z 0.07). The ranges overlap, so neither a single absolute threshold nor a
per-query z-score separates signal from noise on this corpus. The overlap is
not only noise: several "competing" images are the same subject (e.g. the
Claude handbook cover vs its thumbnail), so exact-file ranking is the wrong
success metric here.

A 0.25 floor hid real hits (0.2133, 0.2195, 0.2264) while a noise query still
reached 0.2503. Tightening to 0.251 would be inside measurement noise and would
drop `hexapod assembly` at 0.2507. There is no honest single cutoff, so the
product no longer pretends there is one.

### Decision: a garbage floor plus a separate "Related" group

`min_visual_similarity = 0.20` is now only a **garbage floor**: it removes
vectors that are essentially unrelated to the query. It is not a relevance
classifier. Everything above it that is not backed by text is presented
honestly:

- **Matches** — text hits and both-hits, ordered by reciprocal rank fusion.
  Text is literal evidence, so this list is not capped by the visual threshold.
- **Related by appearance** — visual-only hits above the floor, ordered by
  cosine, capped at `max_visual_results` (default 8). The UI labels the section
  "Related by appearance · approximate" and mutes it; the CLI prints it under
  its own header. A relative margin of 0.05 drops a weak tail that is only
  slightly below the best hit.

This keeps the strong hits (0.21–0.38) visible, shows the weak-but-real Italian
photos instead of hiding them, and stops approximate results from looking
certain. `spiaggia` noise at 0.2383 now appears as a clearly-labelled
suggestion rather than a confident match.

A z-score gate was measured but not adopted: noise reaches z 2.38 while valid
hits fall to z 0.07, so any z threshold either drops real hits or keeps noise.

### Roadmap

The shipped CLIP pair is the limiting factor. A SigLIP 2 image/text model has
markedly better retrieval separation; when it is bundled, re-run this
calibration before changing the floor. Until then, visual search stays
approximate by design.

If the corpus grows, re-run `tools/calibrate_threshold.py` with new pairs and
noise queries before changing the default.
