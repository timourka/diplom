#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date Reader pipeline v5.

Что делает:
  1) ExpDate Products-Real/Products-Synth annotations.json -> crop-ы даты + labels.csv
  2) опционально генерирует реалистичную синтетику на случайных фонах
  3) обучает более мощную TFLite-friendly OCR-модель Conv/TCN + CTC
  4) экспортирует date_reader.tflite и date_reader_meta.json

Быстрый запуск с сильной моделью и синтетикой:
  python date_reader_pipeline.py ^
    --expdate-root D:\\diplom\\Products-Real ^
    --out D:\\diplom\\date_reader_work_v5 ^
    --model large ^
    --input-width 256 ^
    --input-height 48 ^
    --synthetic-count 12000 ^
    --epochs 120 ^
    --patience 25

Зависимости:
  pip install tensorflow pillow opencv-python numpy
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shutil
import string
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageDraw, ImageFont

DEFAULT_ALPHABET = "0123456789./-"
DATE_CLASSES = {"date", "exp"}


def norm_text(text: str, alphabet: str = DEFAULT_ALPHABET) -> str:
    if text is None:
        return ""
    text = str(text).strip().upper()
    text = text.replace("\\", "/").replace("|", "/").replace(" ", "")
    allowed = set(alphabet)
    return "".join(ch for ch in text if ch in allowed)


def looks_like_date(text: str) -> bool:
    digits = re.sub(r"\D", "", text or "")
    return len(digits) >= 4


def safe_stem(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def crop_with_padding(img: Image.Image, bbox, pad: float = 0.25) -> Optional[Image.Image]:
    a, b, c, d = map(float, bbox)
    x1, x2 = sorted([a, c])
    y1, y2 = sorted([b, d])
    w = x2 - x1
    h = y2 - y1
    if w < 2 or h < 2:
        return None
    px = w * pad
    py = h * pad
    W, H = img.size
    x1 = max(0, int(math.floor(x1 - px)))
    y1 = max(0, int(math.floor(y1 - py)))
    x2 = min(W, int(math.ceil(x2 + px)))
    y2 = min(H, int(math.ceil(y2 + py)))
    if x2 <= x1 or y2 <= y1:
        return None
    return img.crop((x1, y1, x2, y2))


def iter_expdate_items(expdate_root: Path) -> Iterable[tuple[Path, dict, str]]:
    for split in ["train", "evaluation", "val", "test"]:
        split_dir = expdate_root / split
        ann_path = split_dir / "annotations.json"
        img_dir = split_dir / "images"
        if ann_path.exists() and img_dir.exists():
            data = json.loads(ann_path.read_text(encoding="utf-8"))
            for img_name, rec in data.items():
                yield img_dir / img_name, rec, split

    ann_path = expdate_root / "annotations.json"
    img_dir = expdate_root / "images"
    if ann_path.exists() and img_dir.exists():
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        for img_name, rec in data.items():
            yield img_dir / img_name, rec, "train"

    if ann_path.exists():
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        for img_name, rec in data.items():
            p = expdate_root / img_name
            if p.exists():
                yield p, rec, "train"


def extract_zip_to_temp(zip_path: Path) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="expdate_zip_"))
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(tmp)
    children = [p for p in tmp.iterdir() if p.is_dir()]
    if len(children) == 1:
        return children[0]
    return tmp


def collect_background_images(source_root: Optional[Path], zip_path: Optional[Path] = None, limit: int = 800) -> list[Path]:
    if zip_path is not None:
        source_root = extract_zip_to_temp(zip_path)
    if source_root is None:
        return []
    paths: list[Path] = []
    for img_path, _, _ in iter_expdate_items(source_root):
        if img_path.exists():
            paths.append(img_path)
            if len(paths) >= limit:
                break
    random.shuffle(paths)
    return paths


def prepare_crops(
    source_root: Optional[Path],
    out_dir: Path,
    zip_path: Optional[Path] = None,
    extra_csv: Optional[Path] = None,
    alphabet: str = DEFAULT_ALPHABET,
    val_ratio: float = 0.12,
    pad: float = 0.25,
    seed: int = 42,
) -> tuple[Path, Path, int]:
    random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_root = out_dir / "crops"
    if crop_root.exists():
        shutil.rmtree(crop_root)
    (crop_root / "train").mkdir(parents=True, exist_ok=True)
    (crop_root / "val").mkdir(parents=True, exist_ok=True)

    if zip_path is not None:
        source_root = extract_zip_to_temp(zip_path)
    if source_root is None:
        raise SystemExit("Нужно указать --expdate-root или --zip")

    rows: list[tuple[str, str]] = []
    count = 0
    skipped = 0

    for img_path, rec, split in iter_expdate_items(source_root):
        if not img_path.exists():
            skipped += 1
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            skipped += 1
            continue
        anns = rec.get("ann", []) or []
        for j, a in enumerate(anns):
            cls = str(a.get("cls", "")).lower()
            if cls not in DATE_CLASSES:
                continue
            text = norm_text(a.get("transcription", ""), alphabet)
            if not looks_like_date(text):
                continue
            bbox = a.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            crop = crop_with_padding(img, bbox, pad=pad)
            if crop is None:
                skipped += 1
                continue
            target_split = "val" if split in {"evaluation", "val", "test"} else "train"
            if target_split == "train" and random.random() < val_ratio:
                target_split = "val"
            name = f"{count:07d}_{safe_stem(img_path.stem)}_{j}.png"
            rel = Path("crops") / target_split / name
            crop.save(out_dir / rel)
            rows.append((rel.as_posix(), text))
            count += 1

    if extra_csv and extra_csv.exists():
        with extra_csv.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                p = Path(r.get("path") or r.get("image") or "")
                text = norm_text(r.get("text") or r.get("label") or "", alphabet)
                if not p.exists() or not looks_like_date(text):
                    continue
                try:
                    crop = Image.open(p).convert("RGB")
                except Exception:
                    continue
                target_split = "val" if random.random() < val_ratio else "train"
                name = f"extra_{count:07d}_{safe_stem(p.stem)}.png"
                rel = Path("crops") / target_split / name
                crop.save(out_dir / rel)
                rows.append((rel.as_posix(), text))
                count += 1

    random.shuffle(rows)
    train_csv = out_dir / "labels_train.csv"
    val_csv = out_dir / "labels_val.csv"
    with train_csv.open("w", encoding="utf-8", newline="") as ft, val_csv.open("w", encoding="utf-8", newline="") as fv:
        wt, wv = csv.writer(ft), csv.writer(fv)
        wt.writerow(["path", "text"])
        wv.writerow(["path", "text"])
        for rel, text in rows:
            if "/val/" in rel.replace("\\", "/"):
                wv.writerow([rel, text])
            else:
                wt.writerow([rel, text])

    print(f"[PREPARE] real crops: {count}, skipped: {skipped}")
    print(f"[PREPARE] train csv: {train_csv}")
    print(f"[PREPARE] val csv: {val_csv}")
    return train_csv, val_csv, count


# -------------------------- Synthetic generation --------------------------

def find_font_paths() -> list[Path]:
    roots = [
        Path("C:/Windows/Fonts"),
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path.home() / ".fonts",
    ]
    out: list[Path] = []
    good_words = ["arial", "calibri", "consola", "cour", "tahoma", "verdana", "dejavu", "liberation", "mono", "ocr", "times", "segui"]
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.ttf"):
            low = p.name.lower()
            if any(w in low for w in good_words):
                out.append(p)
        for p in root.rglob("*.otf"):
            low = p.name.lower()
            if any(w in low for w in good_words):
                out.append(p)
    random.shuffle(out)
    return out[:200]


def random_date_text() -> str:
    year = random.randint(2020, 2032)
    month = random.randint(1, 12)
    day = random.randint(1, 28)
    sep = random.choice([".", "/", "-", ".", "."])
    yy = year % 100
    fmt = random.choices(
        ["dmy4", "ymd4", "dmy2", "compact_dmy", "ym", "my"],
        weights=[45, 22, 12, 8, 8, 5],
        k=1,
    )[0]
    if fmt == "dmy4":
        return f"{day:02d}{sep}{month:02d}{sep}{year:04d}"
    if fmt == "ymd4":
        return f"{year:04d}{sep}{month:02d}{sep}{day:02d}"
    if fmt == "dmy2":
        return f"{day:02d}{sep}{month:02d}{sep}{yy:02d}"
    if fmt == "compact_dmy":
        return f"{day:02d}{month:02d}{year:04d}"
    if fmt == "ym":
        return f"{year:04d}{sep}{month:02d}"
    return f"{month:02d}{sep}{year:04d}"


def make_background(bg_paths: list[Path], w: int, h: int) -> Image.Image:
    if bg_paths and random.random() < 0.85:
        p = random.choice(bg_paths)
        try:
            img = Image.open(p).convert("RGB")
            W, H = img.size
            if W > 4 and H > 4:
                # случайный кроп из упаковки
                crop_w = random.randint(max(4, min(W, w)), max(5, W)) if W > w else W
                crop_h = random.randint(max(4, min(H, h)), max(5, H)) if H > h else H
                x = random.randint(0, max(0, W - crop_w))
                y = random.randint(0, max(0, H - crop_h))
                img = img.crop((x, y, x + crop_w, y + crop_h)).resize((w, h), Image.BILINEAR)
                return img
        except Exception:
            pass
    # fallback: шум/градиент
    base = np.random.normal(random.randint(160, 230), random.randint(5, 30), (h, w, 3)).clip(0, 255).astype("uint8")
    return Image.fromarray(base, "RGB")


def draw_dot_matrix(mask: Image.Image, dot_step: int = 3, dot_radius: int = 1) -> Image.Image:
    src = np.asarray(mask)
    out = Image.new("L", mask.size, 0)
    d = ImageDraw.Draw(out)
    H, W = src.shape
    jitter = max(0, dot_step // 3)
    for y in range(0, H, dot_step):
        for x in range(0, W, dot_step):
            y1, y2 = max(0, y - 1), min(H, y + 2)
            x1, x2 = max(0, x - 1), min(W, x + 2)
            if src[y1:y2, x1:x2].max() > 32 and random.random() > 0.06:
                xx = x + random.randint(-jitter, jitter) if jitter else x
                yy = y + random.randint(-jitter, jitter) if jitter else y
                d.ellipse((xx - dot_radius, yy - dot_radius, xx + dot_radius, yy + dot_radius), fill=random.randint(180, 255))
    return out


def render_synthetic_date(text: str, bg_paths: list[Path], font_paths: list[Path], input_w: int, input_h: int) -> Image.Image:
    # генерируем немного больше входного размера, чтобы имитировать реальные crop-ы разной формы
    w = random.randint(int(input_w * 0.65), int(input_w * 1.35))
    h = random.randint(int(input_h * 0.75), int(input_h * 1.6))
    bg = make_background(bg_paths, w, h).convert("RGB")

    # иногда размываем фон, чтобы дата лучше/хуже читалась как на упаковке
    if random.random() < 0.35:
        bg = bg.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.0)))
    if random.random() < 0.6:
        bg = ImageEnhance.Contrast(bg).enhance(random.uniform(0.75, 1.25))
    if random.random() < 0.6:
        bg = ImageEnhance.Brightness(bg).enhance(random.uniform(0.75, 1.15))

    # подбираем размер шрифта так, чтобы строка влезла
    font = None
    for _ in range(20):
        size = random.randint(max(12, int(h * 0.35)), max(13, int(h * 0.78)))
        if font_paths:
            try:
                font = ImageFont.truetype(str(random.choice(font_paths)), size=size)
            except Exception:
                font = ImageFont.load_default()
        else:
            font = ImageFont.load_default()
        bbox = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if tw <= w * 0.96 and th <= h * 0.9:
            break

    mask = Image.new("L", (w, h), 0)
    dmask = ImageDraw.Draw(mask)
    bbox = dmask.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = random.randint(0, max(0, w - tw - 1))
    y = random.randint(0, max(0, h - th - 1))
    dmask.text((x, y - bbox[1]), text, font=font, fill=255)

    # 65% имитируем точечно-матричную маркировку
    if random.random() < 0.65:
        step = random.choice([2, 3, 3, 4])
        radius = random.choice([1, 1, 2])
        mask = draw_dot_matrix(mask, dot_step=step, dot_radius=radius)
        if random.random() < 0.25:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.15, 0.55)))
    else:
        if random.random() < 0.25:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.7)))

    bg_gray = np.asarray(bg.convert("L")).mean()
    if random.random() < 0.82:
        # чаще тёмная печать
        ink = random.randint(0, 80) if bg_gray > 110 else random.randint(170, 255)
    else:
        ink = random.randint(120, 245)
    fg = Image.new("RGB", (w, h), (ink, ink, ink))
    bg.paste(fg, mask=mask)

    # геометрия/шум
    if random.random() < 0.55:
        bg = bg.rotate(random.uniform(-3.0, 3.0), expand=True, fillcolor=tuple(np.asarray(bg).reshape(-1, 3).mean(axis=0).astype(int)))
    if random.random() < 0.45:
        bg = bg.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 0.8)))
    if random.random() < 0.55:
        arr = np.asarray(bg).astype("int16")
        noise = np.random.normal(0, random.uniform(2, 12), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype("uint8")
        bg = Image.fromarray(arr, "RGB")
    if random.random() < 0.35:
        bg = ImageEnhance.Contrast(bg).enhance(random.uniform(0.7, 1.5))

    return bg


def append_csv_rows(csv_path: Path, rows: list[tuple[str, str]]) -> None:
    exists = csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["path", "text"])
        for row in rows:
            w.writerow(row)


def generate_synthetic_dataset(
    out_dir: Path,
    count: int,
    alphabet: str,
    bg_paths: list[Path],
    input_w: int,
    input_h: int,
    seed: int = 123,
) -> int:
    if count <= 0:
        return 0
    random.seed(seed)
    syn_dir = out_dir / "crops" / "train_synth"
    syn_dir.mkdir(parents=True, exist_ok=True)
    font_paths = find_font_paths()
    rows: list[tuple[str, str]] = []
    for i in range(count):
        text = norm_text(random_date_text(), alphabet)
        img = render_synthetic_date(text, bg_paths, font_paths, input_w=input_w, input_h=input_h)
        name = f"synth_{i:07d}.png"
        rel = Path("crops") / "train_synth" / name
        img.save(out_dir / rel, quality=90)
        rows.append((rel.as_posix(), text))
        if (i + 1) % 1000 == 0:
            print(f"[SYNTH] generated {i + 1}/{count}")
    append_csv_rows(out_dir / "labels_train.csv", rows)
    print(f"[SYNTH] added to train: {len(rows)}")
    return len(rows)


# -------------------------- Training --------------------------

def _import_tf():
    try:
        import tensorflow as tf  # type: ignore
        return tf
    except Exception as e:
        raise SystemExit("TensorFlow не установлен. Установи: pip install tensorflow\n" + str(e))


def read_labels(csv_path: Path) -> list[tuple[str, str]]:
    rows = []
    if not csv_path.exists():
        return rows
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            p = r.get("path", "")
            t = r.get("text", "")
            if p and t:
                rows.append((p, t))
    return rows


def preprocess_pil(path: Path, width: int, height: int, augment: bool = False) -> np.ndarray:
    img = Image.open(path).convert("L")
    if augment:
        if random.random() < 0.55:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.65, 1.9))
        if random.random() < 0.30:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.75, 1.25))
        if random.random() < 0.22:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 0.7)))
        if random.random() < 0.18:
            img = ImageOps.invert(img)
    w, h = img.size
    scale = min(width / max(1, w), height / max(1, h))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    img = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("L", (width, height), color=255)
    canvas.paste(img, (0, (height - nh) // 2))
    arr = np.asarray(canvas).astype("float32") / 255.0
    return arr[..., None]


def encode_text(text: str, alphabet: str, max_len: int) -> np.ndarray:
    table = {ch: i for i, ch in enumerate(alphabet)}
    ids = [table[ch] for ch in text if ch in table]
    ids = ids[:max_len]
    out = np.zeros((max_len,), dtype="int32")
    out[:len(ids)] = ids
    return out


def make_dataset(tf, rows, base_dir: Path, alphabet: str, batch: int, max_len: int, augment: bool, input_w: int, input_h: int, time_steps: int):
    paths = [str(base_dir / p) for p, _ in rows]
    texts = [t for _, t in rows]

    def gen():
        while True:
            idxs = list(range(len(paths)))
            random.shuffle(idxs)
            for i in idxs:
                x = preprocess_pil(Path(paths[i]), width=input_w, height=input_h, augment=augment)
                y = encode_text(texts[i], alphabet, max_len)
                label_len = np.array([min(len(texts[i]), max_len)], dtype="int32")
                input_len = np.array([time_steps], dtype="int32")
                yield {"image": x, "label": y, "input_len": input_len, "label_len": label_len}, np.zeros((1,), dtype="float32")

    sig = (
        {
            "image": tf.TensorSpec(shape=(input_h, input_w, 1), dtype=tf.float32),
            "label": tf.TensorSpec(shape=(max_len,), dtype=tf.int32),
            "input_len": tf.TensorSpec(shape=(1,), dtype=tf.int32),
            "label_len": tf.TensorSpec(shape=(1,), dtype=tf.int32),
        },
        tf.TensorSpec(shape=(1,), dtype=tf.float32),
    )
    return tf.data.Dataset.from_generator(gen, output_signature=sig).batch(batch).prefetch(tf.data.AUTOTUNE)


def conv_bn_relu(tf, x, filters, k=3, name=None):
    x = tf.keras.layers.Conv2D(filters, k, padding="same", use_bias=False, name=None if name is None else name + "_conv")(x)
    x = tf.keras.layers.BatchNormalization(name=None if name is None else name + "_bn")(x)
    return tf.keras.layers.Activation("relu", name=None if name is None else name + "_relu")(x)


def tcn_block(tf, x, filters, dilation, dropout=0.10, name="tcn"):
    shortcut = x
    y = tf.keras.layers.SeparableConv1D(filters, 5, padding="same", dilation_rate=dilation, use_bias=False, name=name + "_sep1")(x)
    y = tf.keras.layers.BatchNormalization(name=name + "_bn1")(y)
    y = tf.keras.layers.Activation("relu", name=name + "_relu1")(y)
    y = tf.keras.layers.Dropout(dropout, name=name + "_drop")(y)
    y = tf.keras.layers.SeparableConv1D(filters, 5, padding="same", dilation_rate=dilation, use_bias=False, name=name + "_sep2")(y)
    y = tf.keras.layers.BatchNormalization(name=name + "_bn2")(y)
    if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv1D(filters, 1, padding="same", name=name + "_proj")(shortcut)
    y = tf.keras.layers.Add(name=name + "_add")([shortcut, y])
    return tf.keras.layers.Activation("relu", name=name + "_out")(y)


def build_models(tf, alphabet: str, input_w: int, input_h: int, max_len: int = 24, model_size: str = "base"):
    num_classes = len(alphabet) + 1
    if model_size == "small":
        channels = [32, 64, 128, 192]
        seq_filters = 192
        tcn_layers = [1, 2]
    elif model_size == "large":
        channels = [64, 128, 192, 256, 384]
        seq_filters = 384
        tcn_layers = [1, 2, 4, 8, 1, 2]
    else:
        channels = [48, 96, 160, 256]
        seq_filters = 256
        tcn_layers = [1, 2, 4, 1]

    image = tf.keras.Input(shape=(input_h, input_w, 1), name="image")
    x = conv_bn_relu(tf, image, channels[0], name="b1a")
    x = conv_bn_relu(tf, x, channels[0], name="b1b")
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)

    x = conv_bn_relu(tf, x, channels[1], name="b2a")
    x = conv_bn_relu(tf, x, channels[1], name="b2b")
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)

    x = conv_bn_relu(tf, x, channels[2], name="b3a")
    x = conv_bn_relu(tf, x, channels[2], name="b3b")
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 1), name="pool3")(x)

    x = conv_bn_relu(tf, x, channels[3], name="b4a")
    x = conv_bn_relu(tf, x, channels[3], name="b4b")
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 1), name="pool4")(x)

    if len(channels) > 4:
        x = conv_bn_relu(tf, x, channels[4], name="b5a")
        x = conv_bn_relu(tf, x, channels[4], name="b5b")

    # Любая высота -> 1, ширина остаётся временной осью.
    x = tf.keras.layers.Lambda(lambda t: tf.reduce_mean(t, axis=1), name="height_mean")(x)  # B,T,C
    x = tf.keras.layers.Conv1D(seq_filters, 1, padding="same", activation="relu", name="seq_proj")(x)
    for idx, dil in enumerate(tcn_layers):
        x = tcn_block(tf, x, seq_filters, dilation=dil, dropout=0.08 if model_size != "large" else 0.12, name=f"tcn{idx+1}")
    logits = tf.keras.layers.Dense(num_classes, name="logits")(x)
    pred_model = tf.keras.Model(image, logits, name=f"date_reader_{model_size}")

    label = tf.keras.Input(shape=(max_len,), dtype="int32", name="label")
    input_len = tf.keras.Input(shape=(1,), dtype="int32", name="input_len")
    label_len = tf.keras.Input(shape=(1,), dtype="int32", name="label_len")

    def ctc_loss_layer(args):
        labels, y_pred, inp_len, lab_len = args
        y_pred = tf.nn.softmax(y_pred, axis=-1)
        return tf.keras.backend.ctc_batch_cost(labels, y_pred, inp_len, lab_len)

    loss = tf.keras.layers.Lambda(ctc_loss_layer, name="ctc_loss")([label, logits, input_len, label_len])
    train_model = tf.keras.Model(inputs=[image, label, input_len, label_len], outputs=loss)
    train_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=lambda y_true, y_pred: y_pred,
        jit_compile=False,
    )
    return train_model, pred_model


def greedy_decode_np(logits: np.ndarray, alphabet: str) -> str:
    ids = logits.argmax(axis=-1)
    if ids.ndim == 2:
        ids = ids[0]
    blank = len(alphabet)
    out = []
    prev = None
    for i in ids.tolist():
        if i != prev and i != blank:
            if 0 <= i < len(alphabet):
                out.append(alphabet[i])
        prev = i
    return "".join(out)


def edit_distance(a: str, b: str) -> int:
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        ndp = [i]
        for j, cb in enumerate(b, 1):
            ndp.append(min(dp[j] + 1, ndp[-1] + 1, dp[j - 1] + (ca != cb)))
        dp = ndp
    return dp[-1]


def evaluate_model(pred_model, rows, work_dir: Path, alphabet: str, input_w: int, input_h: int, limit: int = 300):
    if not rows:
        return 0.0, 1.0
    sample = rows[: min(limit, len(rows))]
    exact = 0
    dist = 0
    chars = 0
    examples = []
    for rel, gt in sample:
        x = preprocess_pil(work_dir / rel, width=input_w, height=input_h, augment=False)[None, ...]
        logits = pred_model.predict(x, verbose=0)
        pred = greedy_decode_np(logits, alphabet)
        if pred == gt:
            exact += 1
        dist += edit_distance(gt, pred)
        chars += max(1, len(gt))
        if len(examples) < 10:
            examples.append((gt, pred))
    return exact / len(sample), dist / chars, examples


class OCRMetricsCallback:
    def __init__(self, tf, pred_model, val_rows, work_dir: Path, alphabet: str, input_w: int, input_h: int):
        self.tf = tf
        self.pred_model = pred_model
        self.val_rows = val_rows
        self.work_dir = work_dir
        self.alphabet = alphabet
        self.input_w = input_w
        self.input_h = input_h

    def callback(self):
        parent = self
        class _Cb(parent.tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % 5 == 0:
                    acc, cer, _ = evaluate_model(parent.pred_model, parent.val_rows, parent.work_dir, parent.alphabet, parent.input_w, parent.input_h, limit=120)
                    print(f"\n[VAL_DECODE] epoch={epoch+1} exact={acc:.3f} CER={cer:.3f}")
        return _Cb()


def train_and_export(work_dir: Path, alphabet: str, epochs: int, batch: int, max_len: int, input_w: int, input_h: int, model_size: str, patience: int, no_early_stopping: bool):
    tf = _import_tf()
    train_rows = read_labels(work_dir / "labels_train.csv")
    val_rows = read_labels(work_dir / "labels_val.csv")

    train_rows = [(p, norm_text(t, alphabet)) for p, t in train_rows if (work_dir / p).exists() and looks_like_date(norm_text(t, alphabet))]
    val_rows = [(p, norm_text(t, alphabet)) for p, t in val_rows if (work_dir / p).exists() and looks_like_date(norm_text(t, alphabet))]

    if not train_rows:
        raise SystemExit("Нет train crop-ов. Пересобери датасет без --skip-prepare")
    if not val_rows:
        val_rows = train_rows[: max(1, len(train_rows) // 10)]

    random.shuffle(train_rows)
    random.shuffle(val_rows)
    print(f"[TRAIN] train rows: {len(train_rows)} | val rows: {len(val_rows)}")
    print(f"[TRAIN] model={model_size}, input={input_w}x{input_h}, alphabet='{alphabet}'")

    train_model, pred_model = build_models(tf, alphabet, input_w=input_w, input_h=input_h, max_len=max_len, model_size=model_size)
    time_steps = int(pred_model.output_shape[1])
    print(f"[TRAIN] time_steps={time_steps}, classes={len(alphabet)+1}")

    train_ds = make_dataset(tf, train_rows, work_dir, alphabet, batch, max_len, augment=True, input_w=input_w, input_h=input_h, time_steps=time_steps)
    val_ds = make_dataset(tf, val_rows, work_dir, alphabet, batch, max_len, augment=False, input_w=input_w, input_h=input_h, time_steps=time_steps)
    steps_per_epoch = max(1, math.ceil(len(train_rows) / batch))
    validation_steps = max(1, math.ceil(len(val_rows) / batch))

    ckpt = work_dir / "best_date_reader.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(ckpt), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=max(4, patience // 4), factor=0.5, min_lr=1e-6),
        OCRMetricsCallback(tf, pred_model, val_rows, work_dir, alphabet, input_w, input_h).callback(),
    ]
    if not no_early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True))

    train_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
    )

    acc, cer, examples = evaluate_model(pred_model, val_rows, work_dir, alphabet, input_w, input_h, limit=400)
    print(f"\n[FINAL] val exact={acc:.3f}, CER={cer:.3f}")
    print("[CHECK] sample predictions:")
    for gt, pred in examples:
        print(f"  GT={gt:14s} PRED={pred}")

    saved = work_dir / "date_reader_saved_model"
    if saved.exists():
        shutil.rmtree(saved)
    pred_model.export(str(saved))
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved))
    # Float16 меньше, но остаётся достаточно безопасным для Android. Можно выключить через --no-float16.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite = converter.convert()
    out = work_dir / "date_reader.tflite"
    out.write_bytes(tflite)

    meta = {
        "alphabet": alphabet,
        "blank_index": len(alphabet),
        "input_width": input_w,
        "input_height": input_h,
        "time_steps": time_steps,
        "num_classes": len(alphabet) + 1,
        "model_size": model_size,
        "decode": "CTC greedy: argmax per timestep, remove repeats, remove blank=len(alphabet)",
        "val_exact_sample": acc,
        "val_cer_sample": cer,
    }
    (work_dir / "date_reader_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] TFLite saved: {out}")
    print(f"[OK] Meta saved: {work_dir / 'date_reader_meta.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expdate-root", type=Path, default=None, help="Products-Real или Products-Synth")
    ap.add_argument("--zip", type=Path, default=None, help="zip с images + annotations.json, для проверки")
    ap.add_argument("--out", type=Path, required=True, help="рабочая папка")
    ap.add_argument("--extra-csv", type=Path, default=None, help="CSV с личными crop-ами: path,text")
    ap.add_argument("--alphabet", default=DEFAULT_ALPHABET)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=24)
    ap.add_argument("--pad", type=float, default=0.25)
    ap.add_argument("--model", choices=["small", "base", "large"], default="large")
    ap.add_argument("--input-width", type=int, default=256)
    ap.add_argument("--input-height", type=int, default=48)
    ap.add_argument("--synthetic-count", type=int, default=0, help="сколько синтетических crop-ов добавить в train")
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--no-early-stopping", action="store_true")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--skip-prepare", action="store_true")
    args = ap.parse_args()

    if not args.skip_prepare:
        _, _, n = prepare_crops(
            source_root=args.expdate_root,
            zip_path=args.zip,
            out_dir=args.out,
            extra_csv=args.extra_csv,
            alphabet=args.alphabet,
            pad=args.pad,
        )
        bg_paths = collect_background_images(args.expdate_root, args.zip, limit=800)
        generate_synthetic_dataset(
            out_dir=args.out,
            count=args.synthetic_count,
            alphabet=args.alphabet,
            bg_paths=bg_paths,
            input_w=args.input_width,
            input_h=args.input_height,
        )
        if n < 50 and not args.prepare_only:
            print("[WARN] crop-ов очень мало. Для проверки пайплайна можно, для реального OCR мало.")
    if args.prepare_only:
        return
    train_and_export(
        args.out,
        args.alphabet,
        args.epochs,
        args.batch,
        args.max_len,
        args.input_width,
        args.input_height,
        args.model,
        args.patience,
        args.no_early_stopping,
    )


if __name__ == "__main__":
    main()
