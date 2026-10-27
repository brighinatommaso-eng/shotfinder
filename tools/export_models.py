"""Export ShotFinder's CLIP encoders to ONNX.

Dev-only script. It needs the ``[export]`` extra (torch, transformers,
sentence-transformers, onnx, onnxruntime); the shipped app only needs
onnxruntime + tokenizers + numpy + Pillow.

Usage::

    python tools/export_models.py [--out models]

Outputs (in ``--out``, default ``<repo>/models``):

    clip_image_fp32.onnx    pixel_values[N,3,224,224] -> image_embeds[N,512]
    clip_text_fp32.onnx     input_ids[N,L] + attention_mask[N,L] -> text_embeds[N,512]
    tokenizer.json          DistilBERT tokenizer used by the text encoder
    clip_image_int8.onnx    dynamic QInt8, only if quality holds
    clip_text_int8.onnx     dynamic QInt8, only if quality holds

Neither graph normalizes its output: the runtime L2-normalizes embeddings.

The script also prints the fp32 image/text cosine matrix, the per-embedding
agreement between ONNX and the original PyTorch models, and the int8-vs-fp32
comparison over 20 samples per model.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

IMAGE_MODEL = "openai/clip-vit-base-patch32"
TEXT_MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"

IMAGE_FP32 = "clip_image_fp32.onnx"
TEXT_FP32 = "clip_text_fp32.onnx"
IMAGE_INT8 = "clip_image_int8.onnx"
TEXT_INT8 = "clip_text_int8.onnx"
TOKENIZER = "tokenizer.json"

QUERIES = ["spiaggia al tramonto", "beach at sunset", "screenshot di codice"]

# CLIP preprocessing constants (must match extractors/vision.py).
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# --------------------------------------------------------------------------
# test images
# --------------------------------------------------------------------------


def _beach_sunset(size: int = 256) -> Image.Image:
    image = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(image)
    horizon = int(size * 0.62)
    for y in range(horizon):
        t = y / max(1, horizon - 1)
        color = (
            int(250 - 30 * t),
            int(150 + 70 * t),
            int(90 + 80 * t),
        )
        draw.line([(0, y), (size, y)], fill=color)
    for y in range(horizon, size):
        t = (y - horizon) / max(1, size - horizon - 1)
        color = (int(20 + 30 * t), int(90 + 40 * t), int(140 + 50 * t))
        draw.line([(0, y), (size, y)], fill=color)
    radius = size // 10
    cx, cy = int(size * 0.72), int(size * 0.40)
    draw.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius], fill=(255, 235, 150)
    )
    draw.ellipse(
        [cx - radius // 2, cy - radius // 2, cx + radius // 2, cy + radius // 2],
        fill=(255, 250, 210),
    )
    return image


def _code_screenshot(size: int = 256) -> Image.Image:
    image = Image.new("RGB", (size, size), (252, 252, 252))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, size, 24], fill=(40, 44, 52))
    for i, color in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        draw.ellipse([8 + i * 18, 7, 18 + i * 18, 17], fill=color)
    lines = [
        "def main(argv):",
        "    total = 0",
        "    for path in argv:",
        "        total += len(path)",
        "    return total",
        "",
        "if __name__ == '__main__':",
        "    print(main(sys.argv[1:]))",
    ]
    y = 44
    for line in lines:
        draw.text((14, y), line, fill=(60, 60, 70))
        y += 22
    return image


def _forest(size: int = 256) -> Image.Image:
    image = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(image)
    for y in range(size):
        t = y / size
        draw.line(
            [(0, y), (size, y)],
            fill=(int(30 + 60 * t), int(90 + 80 * t), int(40 + 40 * t)),
        )
    for x in (40, 110, 190):
        draw.rectangle([x, 80, x + 16, size], fill=(80, 55, 30))
        draw.ellipse([x - 30, 20, x + 46, 110], fill=(35, 110, 45))
    return image


def _test_images() -> list[Image.Image]:
    return [_beach_sunset(), _code_screenshot(), _forest()]


def _sample_images(count: int) -> list[Image.Image]:
    """Deterministic synthetic images for the quantization comparison."""
    images: list[Image.Image] = []
    for index in range(count):
        rng = np.random.default_rng(index)
        base = tuple(int(v) for v in rng.integers(0, 256, 3))
        image = Image.new("RGB", (256, 256), base)
        draw = ImageDraw.Draw(image)
        for _ in range(4):
            x0, y0 = (int(v) for v in rng.integers(0, 200, 2))
            x1, y1 = x0 + int(rng.integers(20, 80)), y0 + int(rng.integers(20, 80))
            color = tuple(int(v) for v in rng.integers(0, 256, 3))
            if index % 2:
                draw.ellipse([x0, y0, x1, y1], fill=color)
            else:
                draw.rectangle([x0, y0, x1, y1], fill=color)
        draw.text((10, 10), f"sample {index}", fill=(255, 255, 255))
        images.append(image)
    return images


def _sample_texts() -> list[str]:
    return [
        "spiaggia al tramonto",
        "beach at sunset",
        "screenshot di codice",
        "a red car on a road",
        "una macchina rossa",
        "invoice with a total",
        "fattura con il totale",
        "mountains covered in snow",
        "montagne innevate",
        "a cat sleeping on a sofa",
        "un gatto che dorme sul divano",
        "chat message from Alice",
        "messaggio di chat di Alice",
        "a chart with rising revenue",
        "grafico con ricavi in crescita",
        "a screenshot of a terminal",
        "terminale con comandi",
        "sunset over the sea",
        "tramonto sul mare",
        "a forest with tall trees",
    ]


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------


def preprocess_images(images: list[Image.Image]) -> np.ndarray:
    """CLIP ViT-B/32 preprocessing: shortest side 224 bicubic, center crop."""
    batch = np.zeros((len(images), 3, 224, 224), dtype=np.float32)
    mean = np.asarray(CLIP_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(CLIP_STD, dtype=np.float32).reshape(3, 1, 1)
    for index, image in enumerate(images):
        image = image.convert("RGB")
        width, height = image.size
        scale = 224 / min(width, height)
        resized = image.resize(
            (max(224, round(width * scale)), max(224, round(height * scale))),
            Image.BICUBIC,
        )
        left = (resized.width - 224) // 2
        top = (resized.height - 224) // 2
        crop = resized.crop((left, top, left + 224, top + 224))
        array = np.asarray(crop, dtype=np.float32) / 255.0
        array = (array.transpose(2, 0, 1) - mean) / std
        batch[index] = array
    return batch


def l2_normalize(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.clip(norms, 1e-12, None)


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = l2_normalize(left)
    right = l2_normalize(right)
    return np.sum(left * right, axis=-1)


# --------------------------------------------------------------------------
# wrappers
# --------------------------------------------------------------------------


class ImageEncoder(torch.nn.Module):
    def __init__(self, clip: torch.nn.Module) -> None:
        super().__init__()
        self.clip = clip

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # transformers 5 returns a BaseModelOutputWithPooling whose
        # pooler_output is already passed through visual_projection.
        return self.clip.get_image_features(pixel_values=pixel_values).pooler_output


class TextEncoder(torch.nn.Module):
    """DistilBERT + masked mean pooling + Dense(768 -> 512)."""

    def __init__(self, bert: torch.nn.Module, dense: torch.nn.Linear) -> None:
        super().__init__()
        self.bert = bert
        self.dense = dense

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts
        return self.dense(pooled)


def _export(
    module: torch.nn.Module,
    dummy: tuple[torch.Tensor, ...],
    path: Path,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
) -> None:
    module.eval()
    torch.onnx.export(
        module,
        dummy,
        str(path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )


# --------------------------------------------------------------------------
# sessions
# --------------------------------------------------------------------------


def _session(path: Path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, (torch.get_num_threads() or 2) // 2)
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def _run_image(session, batch: np.ndarray) -> np.ndarray:
    return session.run(["image_embeds"], {"pixel_values": batch})[0]


def _run_text(session, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    return session.run(
        ["text_embeds"],
        {"input_ids": input_ids, "attention_mask": attention_mask},
    )[0]


def _tokenize(
    tokenizer, texts: list[str], max_length: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )
    return (
        encoded["input_ids"].astype(np.int64),
        encoded["attention_mask"].astype(np.int64),
    )


def _quantize(source: Path, target: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        str(source),
        str(target),
        weight_type=QuantType.QInt8,
    )


def _size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def _best_ms(callable_, runs: int = 5) -> float:
    """Fastest of several runs: robust to scheduler noise from other apps."""
    samples: list[float] = []
    for _ in range(runs):
        started = time.perf_counter()
        callable_()
        samples.append((time.perf_counter() - started) * 1000)
    return min(samples)


def _probe_fp16(
    source: Path, target: Path, sample_pixels: np.ndarray, reference: np.ndarray
) -> tuple[float, np.ndarray, float]:
    """Convert to fp16 and measure size, agreement and CPU latency."""
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    model = onnx.load(str(source))
    onnx.save(convert_float_to_float16(model, keep_io_types=True), str(target))
    session = _session(target)
    cosine = cosine_rows(_run_image(session, sample_pixels), reference)

    single = preprocess_images([_beach_sunset()])
    for _ in range(2):
        _run_image(session, single)
    elapsed = _best_ms(lambda: _run_image(session, single))
    return _size_mb(target), cosine, elapsed


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "models",
        help="output directory (default: <repo>/models)",
    )
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    np.random.seed(0)

    print("== loading PyTorch models ==")
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, CLIPModel

    clip = CLIPModel.from_pretrained(IMAGE_MODEL).eval()
    st = SentenceTransformer(TEXT_MODEL)
    st.eval()
    bert = st[0].model
    dense = st[2].linear
    slow_tokenizer = st.tokenizer
    fast_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, use_fast=True)
    print(f"text max_seq_length = {st.max_seq_length}")

    # -- image export ------------------------------------------------------
    print("\n== exporting image encoder ==")
    image_dummy = torch.zeros(2, 3, 224, 224, dtype=torch.float32)
    _export(
        ImageEncoder(clip),
        (image_dummy,),
        out / IMAGE_FP32,
        ["pixel_values"],
        ["image_embeds"],
        {"pixel_values": {0: "batch"}, "image_embeds": {0: "batch"}},
    )

    # -- text export -------------------------------------------------------
    print("== exporting text encoder ==")
    text_dummy = torch.ones(2, 16, dtype=torch.int64)
    text_dummy_mask = torch.ones(2, 16, dtype=torch.int64)
    _export(
        TextEncoder(bert, dense),
        (text_dummy, text_dummy_mask),
        out / TEXT_FP32,
        ["input_ids", "attention_mask"],
        ["text_embeds"],
        {
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "text_embeds": {0: "batch"},
        },
    )
    fast_tokenizer.save_pretrained(out)
    if not (out / TOKENIZER).exists():
        raise RuntimeError("tokenizer.json was not written")
    for extra in ("tokenizer_config.json", "special_tokens_map.json"):
        (out / extra).unlink(missing_ok=True)
    print(f"saved {TOKENIZER} ({_size_mb(out / TOKENIZER):.2f} MB)")

    # -- fp32 verification -------------------------------------------------
    print("\n== fp32 ONNX vs PyTorch ==")
    images = _test_images()
    pixel_values = preprocess_images(images)

    image_session = _session(out / IMAGE_FP32)
    text_session = _session(out / TEXT_FP32)

    onnx_image = _run_image(image_session, pixel_values)
    with torch.no_grad():
        torch_image = (
            clip.get_image_features(pixel_values=torch.from_numpy(pixel_values))
            .pooler_output.cpu()
            .numpy()
        )
    image_cos = cosine_rows(onnx_image, torch_image)
    print(
        "image per-embedding cosine: "
        + ", ".join(f"{value:.6f}" for value in image_cos)
        + f"  (min {image_cos.min():.6f})"
    )

    input_ids, attention_mask = _tokenize(slow_tokenizer, QUERIES)
    onnx_text = _run_text(text_session, input_ids, attention_mask)
    torch_text = st.encode(QUERIES, normalize_embeddings=False, convert_to_numpy=True)
    text_cos = cosine_rows(onnx_text, torch_text)
    print(
        "text  per-embedding cosine: "
        + ", ".join(f"{value:.6f}" for value in text_cos)
        + f"  (min {text_cos.min():.6f})"
    )

    fast_ids, fast_mask = _tokenize(fast_tokenizer, QUERIES)
    tokenizer_ok = np.array_equal(input_ids, fast_ids) and np.array_equal(
        attention_mask, fast_mask
    )
    print(f"fast tokenizer ids match slow tokenizer: {tokenizer_ok}")

    if image_cos.min() <= 0.999 or text_cos.min() <= 0.999:
        print("ALIGNMENT FAILED: ONNX and PyTorch disagree (min cosine <= 0.999)")
        return 1

    # -- alignment matrix --------------------------------------------------
    print("\n== fp32 cosine similarity matrix (rows=images, cols=queries) ==")
    image_embeds = l2_normalize(onnx_image)
    text_embeds = l2_normalize(onnx_text)
    matrix = image_embeds @ text_embeds.T
    row_labels = ["beach sunset", "code screenshot", "forest"]
    header = f"{'':18}" + "".join(f"{label[:22]:>24}" for label in QUERIES)
    print(header)
    for label, row in zip(row_labels, matrix, strict=True):
        print(f"{label:18}" + "".join(f"{value:24.4f}" for value in row))

    best = matrix.argmax(axis=1)
    print(
        "best query per image: "
        + ", ".join(f"{row_labels[i]} -> {QUERIES[j]}" for i, j in enumerate(best))
    )

    # -- quantization ------------------------------------------------------
    print("\n== dynamic QInt8 quantization ==")
    _quantize(out / IMAGE_FP32, out / IMAGE_INT8)
    _quantize(out / TEXT_FP32, out / TEXT_INT8)

    int8_image_session = _session(out / IMAGE_INT8)
    int8_text_session = _session(out / TEXT_INT8)

    sample_images = _sample_images(20)
    sample_pixels = preprocess_images(sample_images)
    int8_image = _run_image(int8_image_session, sample_pixels)
    int8_image_cos = cosine_rows(
        int8_image, _run_image(image_session, sample_pixels)
    )
    print(
        "image int8 vs fp32 (20 samples): "
        f"mean cosine {int8_image_cos.mean():.6f}, min {int8_image_cos.min():.6f}"
    )

    sample_texts = _sample_texts()
    ids, mask = _tokenize(slow_tokenizer, sample_texts)
    int8_text = _run_text(int8_text_session, ids, mask)
    fp32_text = _run_text(text_session, ids, mask)
    int8_text_cos = cosine_rows(int8_text, fp32_text)
    print(
        "text  int8 vs fp32 (20 samples): "
        f"mean cosine {int8_text_cos.mean():.6f}, min {int8_text_cos.min():.6f}"
    )

    keep_image_int8 = bool(int8_image_cos.mean() >= 0.98)
    keep_text_int8 = bool(int8_text_cos.mean() >= 0.98)
    if not keep_image_int8:
        # int8 loses too much on the vision tower. fp16 keeps the embeddings
        # intact but CPU Execution Provider has no fp16 kernels: it casts to
        # fp32 at every node. Measure both before choosing fp32.
        fp16_path = out / "clip_image_fp16.onnx"
        size_mb, fp16_cos, fp16_ms = _probe_fp16(
            out / IMAGE_FP32,
            fp16_path,
            sample_pixels,
            _run_image(image_session, sample_pixels),
        )
        print(
            f"image fp16 probe: {size_mb:.2f} MB, "
            f"mean cosine {fp16_cos.mean():.6f}, {fp16_ms:.1f} ms/image"
        )
        print("image: mean cosine < 0.98, keeping fp32 (fp16 is much slower on CPU)")
        fp16_path.unlink(missing_ok=True)
        (out / IMAGE_INT8).unlink(missing_ok=True)
    if not keep_text_int8:
        print("text: mean cosine < 0.98, dropping int8 (keep fp32)")
        (out / TEXT_INT8).unlink(missing_ok=True)

    # -- timing ------------------------------------------------------------
    # Release every session first: several live ORT thread pools contend and
    # distort latency measurements.
    import gc

    del int8_image_session, int8_text_session, text_session, image_session
    gc.collect()
    final_image = _session(out / IMAGE_FP32)
    final_text = _session(out / (TEXT_INT8 if keep_text_int8 else TEXT_FP32))

    print("\n== timing (batch 1 and batch 8, fastest of 5) ==")
    single = preprocess_images([_beach_sunset()])
    batch = preprocess_images(_sample_images(8))
    for _ in range(2):
        _run_image(final_image, single)
    elapsed = _best_ms(lambda: _run_image(final_image, single))
    print(f"image encoder (fp32): {elapsed:.1f} ms/image")

    for _ in range(2):
        _run_image(final_image, batch)
    elapsed = _best_ms(lambda: _run_image(final_image, batch))
    print(f"image encoder (fp32, batch 8): {elapsed:.1f} ms ({elapsed / 8:.1f} ms/image)")

    for _ in range(2):
        _run_text(final_text, input_ids[:1], attention_mask[:1])
    elapsed = _best_ms(
        lambda: _run_text(final_text, input_ids[:1], attention_mask[:1])
    )
    label = "int8" if keep_text_int8 else "fp32"
    print(f"text encoder ({label}):  {elapsed:.1f} ms/query")

    # -- sizes -------------------------------------------------------------
    print("\n== files ==")
    for name in (IMAGE_FP32, TEXT_FP32, IMAGE_INT8, TEXT_INT8, TOKENIZER):
        path = out / name
        if path.exists():
            print(f"{name:24} {_size_mb(path):8.2f} MB")
        else:
            print(f"{name:24} (absent)")

    print("\n== verdict ==")
    print(f"image int8 kept: {keep_image_int8}")
    print(f"text  int8 kept: {keep_text_int8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
