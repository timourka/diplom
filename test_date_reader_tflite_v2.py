from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import tensorflow as tf
except Exception as e:
    raise SystemExit("TensorFlow is required: pip install tensorflow pillow numpy") from e


def norm_text(text: str, alphabet: str) -> str:
    text = (text or "").strip().upper()
    text = text.replace("\\", "/").replace("|", "/").replace(" ", "")
    allowed = set(alphabet)
    return "".join(ch for ch in text if ch in allowed)


def preprocess_image(path: Path, width: int, height: int) -> np.ndarray:
    img = Image.open(path).convert("L")
    w, h = img.size
    scale = min(width / max(1, w), height / max(1, h))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("L", (width, height), color=255)
    canvas.paste(img, (0, (height - nh) // 2))
    arr = np.asarray(canvas).astype("float32") / 255.0
    return arr[None, ..., None]


def greedy_decode(logits: np.ndarray, alphabet: str) -> str:
    ids = logits.argmax(axis=-1)
    if ids.ndim == 2:
        ids = ids[0]
    blank = len(alphabet)
    out = []
    prev = None
    for idx in ids.tolist():
        if idx != prev and idx != blank and 0 <= idx < len(alphabet):
            out.append(alphabet[idx])
        prev = idx
    return "".join(out)


def edit_distance(a: str, b: str) -> int:
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        ndp = [i]
        for j, cb in enumerate(b, 1):
            ndp.append(min(
                dp[j] + 1,
                ndp[j - 1] + 1,
                dp[j - 1] + (ca != cb),
            ))
        dp = ndp
    return dp[-1]


def load_rows(labels: Path, base_dir: Path) -> list[tuple[Path, str, str]]:
    rows = []
    with labels.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rel = r.get("image") or r.get("path") or r.get("file") or r.get("filename")
            text = r.get("text") or r.get("label") or r.get("transcription") or ""
            if not rel:
                continue
            p = Path(rel)
            if not p.is_absolute():
                p = base_dir / p
            rows.append((p, text, rel))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--labels", type=Path, help="CSV with image/text or path/text")
    ap.add_argument("--base-dir", type=Path, default=None, help="Base dir for relative image paths in CSV")
    ap.add_argument("--images-dir", type=Path, help="Predict all images in folder without GT")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--show-all", action="store_true")
    args = ap.parse_args()

    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    alphabet = meta.get("alphabet", "0123456789./-")
    width = int(meta.get("input_width", 256))
    height = int(meta.get("input_height", 48))

    interpreter = tf.lite.Interpreter(model_path=str(args.model))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]

    if args.labels:
        base_dir = args.base_dir or args.labels.parent
        rows = load_rows(args.labels, base_dir)
    elif args.images_dir:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        rows = [(p, "", p.name) for p in sorted(args.images_dir.rglob("*")) if p.suffix.lower() in exts]
    else:
        raise SystemExit("Use --labels or --images-dir")

    if args.limit and args.limit > 0:
        rows = rows[:args.limit]

    total = 0
    exact = 0
    cer_sum = 0.0
    errors = []

    for p, raw_gt, rel in rows:
        if not p.exists():
            errors.append((rel, raw_gt, "<missing file>", 1.0))
            continue
        x = preprocess_image(p, width, height)
        interpreter.set_tensor(input_index, x)
        interpreter.invoke()
        logits = interpreter.get_tensor(output_index)
        pred = greedy_decode(logits, alphabet)

        if args.labels:
            gt = norm_text(raw_gt, alphabet)
            ed = edit_distance(gt, pred)
            cer = ed / max(1, len(gt))
            total += 1
            exact += int(pred == gt)
            cer_sum += cer
            if args.show_all or pred != gt:
                errors.append((rel, gt, pred, cer))
        else:
            print(f"{rel}: {pred}")

    if args.labels:
        print(f"TOTAL: {total}")
        print(f"EXACT: {exact / max(1, total):.3f}")
        print(f"CER:   {cer_sum / max(1, total):.3f}")
        print("\nОшибки/примеры:")
        for rel, gt, pred, cer in errors[:50]:
            print(f"{rel}: GT={gt} | PRED={pred} | CER={cer:.3f}")


if __name__ == "__main__":
    main()
