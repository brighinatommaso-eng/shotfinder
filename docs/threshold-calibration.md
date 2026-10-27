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

Decision: keep `min_visual_similarity = 0.25`.

- It rejects the original failure mode: `spiaggia` noise at 0.21–0.24
  (top false positive 0.2383).
- It keeps the strong visual hits (0.25–0.38).
- One measured noise query reaches 0.2503, 0.0003 above the floor; separating it
  would require 0.251, which is within measurement noise and would drop the
  `hexapod assembly` hit at 0.2507.
- The two weak Italian OCR-heavy photos (0.2195, 0.2264) are better served by
  text search, which already finds them via OCR.

A z-score gate was measured but not adopted: noise reaches z 2.38 while valid
hits fall to z 0.07, so any z threshold either drops real hits or keeps noise.
If the corpus grows, re-run the tool with new pairs and noise queries before
changing the default.
