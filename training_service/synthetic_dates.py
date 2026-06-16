from __future__ import annotations

import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFilter, ImageFont

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SYNTHETIC_TRAIN_IMAGES = "synthetic_dates/images"
SYNTHETIC_TRAIN_LABELS = "synthetic_dates/labels"

ALLOWED_ALPHABET = "годен до1234567890:,.-/ exp"

# Background clutter is intentionally wider than the target date alphabet.
# These strings imitate packaging text that is NOT annotated as an expiry date.
# This helps the detector learn that not every printed character block is the date.
BACKGROUND_WORDS = [
    "состав", "масса", "нетто", "белки", "жиры", "углеводы", "ккал", "энергия",
    "хранить", "при температуре", "условия хранения", "после вскрытия", "партия",
    "изготовитель", "изготовлено", "упаковано", "адрес", "гост", "ту", "еас",
    "молоко", "кефир", "йогурт", "сыр", "творог", "шоколад", "печенье", "соус",
    "product", "storage", "nutrition", "ingredients", "batch", "lot", "net wt", "made in",
    "barcode", "scan", "quality", "keep cool", "best before", "packed",
]
BACKGROUND_ALPHABET = (
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789.,:;/-+()%№ "
)


@dataclass(frozen=True)
class SyntheticDatesResult:
    enabled: bool
    real_train_images: int
    generated_images: int
    target_share: float
    train_entry: str | None = None

    @property
    def total_train_images(self) -> int:
        return self.real_train_images + self.generated_images

    @property
    def actual_share(self) -> float:
        total = self.total_train_images
        return 0.0 if total <= 0 else self.generated_images / total


def add_synthetic_dates_to_training_set(
    dataset_dir: Path,
    data_yaml_path: Path,
    *,
    imgsz: int,
    target_share: float = 1.0 / 3.0,
    enabled: bool = True,
    max_images: int | None = None,
    seed: int | str | None = None,
) -> SyntheticDatesResult:
    """Generate synthetic expiry-date samples and add them to train only.

    The backend API is not involved. The function only changes the unpacked local
    YOLO dataset before the training call:
      - synthetic images go to synthetic_dates/images;
      - labels go to synthetic_dates/labels;
      - dataset.yaml train receives an extra entry: synthetic_dates/images;
      - val is not changed, so synthetic samples do not leak into validation.
    """

    dataset_dir = dataset_dir.resolve()
    data_yaml_path = data_yaml_path.resolve()
    data = _read_dataset_yaml(data_yaml_path)

    real_train_images = _count_train_images(dataset_dir, data)
    if not enabled:
        _remove_synthetic_entry_if_present(data)
        _write_dataset_yaml(data_yaml_path, data)
        return SyntheticDatesResult(False, real_train_images, 0, target_share)

    target_share = _normalize_share(target_share)
    real_train_images = max(0, real_train_images)
    if real_train_images == 0:
        return SyntheticDatesResult(True, 0, 0, target_share)

    synthetic_count = math.ceil(real_train_images * target_share / (1.0 - target_share))
    if max_images is not None and max_images > 0:
        synthetic_count = min(synthetic_count, max_images)

    synth_images_dir = dataset_dir / SYNTHETIC_TRAIN_IMAGES
    synth_labels_dir = dataset_dir / SYNTHETIC_TRAIN_LABELS
    if synth_images_dir.exists():
        shutil.rmtree(synth_images_dir)
    if synth_labels_dir.exists():
        shutil.rmtree(synth_labels_dir)
    synth_images_dir.mkdir(parents=True, exist_ok=True)
    synth_labels_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(_seed_to_int(seed))
    generator = SyntheticDateImageGenerator(image_size=max(64, int(imgsz)), rng=rng)

    generated = 0
    for index in range(1, synthetic_count + 1):
        image, bbox = generator.generate()
        stem = f"synth_date_{index:06d}"
        image_path = synth_images_dir / f"{stem}.jpg"
        label_path = synth_labels_dir / f"{stem}.txt"

        image.save(image_path, "JPEG", quality=rng.randint(70, 95), optimize=True)
        label_path.write_text(_bbox_to_yolo_line(bbox, image.width, image.height), encoding="utf-8")
        generated += 1

    _ensure_synthetic_train_entry(data)
    _write_dataset_yaml(data_yaml_path, data)
    _write_meta(dataset_dir, real_train_images, generated, target_share, imgsz)

    return SyntheticDatesResult(
        enabled=True,
        real_train_images=real_train_images,
        generated_images=generated,
        target_share=target_share,
        train_entry=SYNTHETIC_TRAIN_IMAGES,
    )


class SyntheticDateImageGenerator:
    def __init__(self, *, image_size: int, rng: random.Random) -> None:
        self.image_size = int(image_size)
        self.rng = rng
        self.fonts = _load_fonts()

    def generate(self) -> tuple[Image.Image, tuple[int, int, int, int]]:
        for _ in range(60):
            background = self._make_background()
            text = self._make_text()
            layer = self._render_text_layer(text)
            layer = self._distort_text_layer(layer)

            bbox_on_layer = _alpha_bbox(layer)
            if bbox_on_layer is None:
                continue

            layer_w, layer_h = layer.size
            if layer_w >= self.image_size or layer_h >= self.image_size:
                continue

            max_x = self.image_size - layer_w
            max_y = self.image_size - layer_h
            x = self.rng.randint(0, max(0, max_x))
            y = self.rng.randint(0, max(0, max_y))

            composed = background.convert("RGBA")
            composed.alpha_composite(layer, (x, y))
            composed = composed.convert("RGB")
            composed = self._postprocess(composed)

            x1 = x + bbox_on_layer[0]
            y1 = y + bbox_on_layer[1]
            x2 = x + bbox_on_layer[2]
            y2 = y + bbox_on_layer[3]
            bbox = _clip_bbox((x1, y1, x2, y2), self.image_size, self.image_size)
            if _bbox_is_reasonable(bbox, self.image_size, self.image_size):
                return composed, bbox

        # Very conservative fallback; should rarely be used.
        background = self._make_background().convert("RGB")
        bbox = (self.image_size // 5, self.image_size // 3, self.image_size * 4 // 5, self.image_size // 2)
        draw = ImageDraw.Draw(background)
        font = self._choose_font(max(14, self.image_size // 14))
        draw.text((bbox[0], bbox[1]), "годен до 12.06.26", font=font, fill=(25, 25, 25))
        return background, bbox

    def _make_background(self) -> Image.Image:
        mode = self.rng.choice(["solid", "linear_gradient", "radial_gradient", "soft_stripes"])
        base = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)

        c1 = np.array(_random_packaging_color(self.rng), dtype=np.float32)
        c2 = np.array(_near_color(c1, self.rng), dtype=np.float32)

        if mode == "solid":
            base[:, :] = c1
        elif mode == "linear_gradient":
            axis = self.rng.choice([0, 1])
            t = np.linspace(0.0, 1.0, self.image_size, dtype=np.float32)
            if axis == 0:
                t = t.reshape(self.image_size, 1, 1)
            else:
                t = t.reshape(1, self.image_size, 1)
            base = c1 * (1.0 - t) + c2 * t
        elif mode == "radial_gradient":
            yy, xx = np.mgrid[0:self.image_size, 0:self.image_size]
            cx = self.rng.uniform(0.2, 0.8) * self.image_size
            cy = self.rng.uniform(0.2, 0.8) * self.image_size
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            t = np.clip(dist / (self.image_size * self.rng.uniform(0.55, 0.95)), 0.0, 1.0)[..., None]
            base = c1 * (1.0 - t) + c2 * t
        else:
            base[:, :] = c1
            stripe_color = _near_color(c1, self.rng, distance=35)
            step = self.rng.randint(max(8, self.image_size // 18), max(14, self.image_size // 8))
            stripe_w = self.rng.randint(2, max(3, step // 4))
            for x in range(-self.image_size, self.image_size * 2, step):
                x0 = x + self.rng.randint(-3, 3)
                x1 = x0 + stripe_w
                if self.rng.random() < 0.5:
                    base[:, max(0, x0):min(self.image_size, x1)] = stripe_color
                else:
                    for k in range(stripe_w):
                        xx = x0 + k
                        if 0 <= xx < self.image_size:
                            yy = np.arange(self.image_size)
                            base[yy, np.clip(xx + yy // max(8, step), 0, self.image_size - 1)] = stripe_color

        noise_level = self.rng.uniform(2.0, 12.0)
        base += self.rng.normalvariate(0, noise_level) * np.ones_like(base)
        base += np.random.default_rng(self.rng.randint(0, 2**31 - 1)).normal(0, noise_level, base.shape)
        base = np.clip(base, 0, 255).astype(np.uint8)
        image = Image.fromarray(base, mode="RGB")
        if self.rng.random() < 0.88:
            image = self._add_background_letter_clutter(image)
        if self.rng.random() < 0.35:
            image = image.filter(ImageFilter.GaussianBlur(radius=self.rng.uniform(0.15, 0.65)))
        return image

    def _add_background_letter_clutter(self, image: Image.Image) -> Image.Image:
        """Draw unlabelled packaging-like letters behind the target date.

        These elements are deliberately not date labels. They make synthetic
        samples closer to real packages where the date is surrounded by product
        text, nutrition tables, barcodes, lot numbers and decorative typography.
        """

        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        groups = self.rng.randint(2, 8)
        for _ in range(groups):
            style = self.rng.choice([
                "word", "word", "word", "small_lines", "nutrition", "lot_code", "barcode", "letters_cloud"
            ])
            if style == "barcode":
                self._draw_background_barcode(layer)
                continue

            text = self._make_background_text(style)
            if not text:
                continue

            font_size = self._background_font_size(style)
            font = self._choose_font(font_size)
            opacity = self.rng.randint(22, 115)
            if style in {"small_lines", "nutrition"}:
                opacity = self.rng.randint(32, 135)
            color = self._background_text_color(opacity)

            text_layer = self._render_background_text_layer(text, font, color, style)
            if text_layer.width <= 2 or text_layer.height <= 2:
                continue

            if self.rng.random() < 0.70:
                angle = self.rng.uniform(-18, 18)
                if style == "letters_cloud":
                    angle = self.rng.uniform(-35, 35)
                text_layer = text_layer.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)

            if self.rng.random() < 0.18:
                text_layer = text_layer.filter(ImageFilter.GaussianBlur(radius=self.rng.uniform(0.15, 0.75)))

            x = self.rng.randint(-text_layer.width // 4, max(0, self.image_size - text_layer.width // 2))
            y = self.rng.randint(-text_layer.height // 4, max(0, self.image_size - text_layer.height // 2))
            layer.alpha_composite(text_layer, (x, y))

        # Add a few thin packaging/table lines so backgrounds are less sterile.
        if self.rng.random() < 0.55:
            draw = ImageDraw.Draw(layer)
            line_color = self._background_text_color(self.rng.randint(18, 70))
            for _ in range(self.rng.randint(2, 9)):
                if self.rng.random() < 0.5:
                    y = self.rng.randint(0, self.image_size - 1)
                    draw.line((0, y, self.image_size, y + self.rng.randint(-4, 4)), fill=line_color, width=self.rng.choice([1, 1, 2]))
                else:
                    x = self.rng.randint(0, self.image_size - 1)
                    draw.line((x, 0, x + self.rng.randint(-4, 4), self.image_size), fill=line_color, width=self.rng.choice([1, 1, 2]))

        composed = image.convert("RGBA")
        composed.alpha_composite(layer)
        return composed.convert("RGB")

    def _make_background_text(self, style: str) -> str:
        if style == "small_lines":
            lines = []
            for _ in range(self.rng.randint(2, 5)):
                word = self.rng.choice(BACKGROUND_WORDS)
                value = self.rng.choice([
                    f"{self.rng.randint(1, 99)} г",
                    f"{self.rng.randint(1, 99)}%",
                    f"{self.rng.randint(100, 999)} ккал",
                    f"{self.rng.randint(0, 30)}..{self.rng.randint(1, 25)} c",
                    "см. упаковку",
                    "без гмо",
                    "eac",
                ])
                lines.append(f"{word}: {value}")
            return "\n".join(lines)

        if style == "nutrition":
            rows = [
                f"белки {self.rng.randint(0, 20)},{self.rng.randint(0, 9)} г",
                f"жиры {self.rng.randint(0, 30)},{self.rng.randint(0, 9)} г",
                f"углеводы {self.rng.randint(0, 80)},{self.rng.randint(0, 9)} г",
                f"ккал {self.rng.randint(80, 590)}",
            ]
            self.rng.shuffle(rows)
            return "\n".join(rows[: self.rng.randint(2, 4)])

        if style == "lot_code":
            # Not formatted like a date on purpose: background negatives should be text-like,
            # but should not create too many unlabelled date stamps.
            chunks = [
                self.rng.choice(["lot", "партия", "batch", "код", "арт"]),
                "".join(self.rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(self.rng.randint(1, 3))),
                "".join(str(self.rng.randint(0, 9)) for _ in range(self.rng.randint(3, 7))),
            ]
            return " ".join(chunks)

        if style == "letters_cloud":
            return "".join(self.rng.choice(BACKGROUND_ALPHABET) for _ in range(self.rng.randint(12, 45))).strip()

        word_count = self.rng.randint(1, 5)
        text = " ".join(self.rng.choice(BACKGROUND_WORDS) for _ in range(word_count))
        if self.rng.random() < 0.35:
            text += f" {self.rng.randint(10, 999)}"
        return text

    def _background_font_size(self, style: str) -> int:
        if style in {"small_lines", "nutrition"}:
            return self.rng.randint(max(7, self.image_size // 42), max(11, self.image_size // 22))
        if style == "letters_cloud":
            return self.rng.randint(max(9, self.image_size // 34), max(18, self.image_size // 12))
        return self.rng.randint(max(8, self.image_size // 32), max(20, self.image_size // 9))

    def _background_text_color(self, alpha: int) -> tuple[int, int, int, int]:
        rgb = self.rng.choice([
            (0, 0, 0), (35, 35, 35), (80, 80, 80), (255, 255, 255),
            (30, 60, 120), (120, 35, 35), (35, 110, 70), (130, 80, 20),
        ])
        return rgb[0], rgb[1], rgb[2], max(0, min(255, alpha))

    def _render_background_text_layer(
        self,
        text: str,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        color: tuple[int, int, int, int],
        style: str,
    ) -> Image.Image:
        lines = text.splitlines() or [text]
        dummy = Image.new("L", (8, 8))
        d = ImageDraw.Draw(dummy)
        boxes = [d.textbbox((0, 0), line, font=font, stroke_width=0) for line in lines]
        widths = [max(1, box[2] - box[0]) for box in boxes]
        heights = [max(1, box[3] - box[1]) for box in boxes]
        pad = 4
        line_gap = max(1, int(max(heights) * 0.25)) if heights else 2
        w = min(self.image_size * 2, max(widths) + pad * 2)
        h = min(self.image_size * 2, sum(heights) + line_gap * (len(lines) - 1) + pad * 2)
        layer = Image.new("RGBA", (max(2, w), max(2, h)), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)

        y = pad
        for line, box, height in zip(lines, boxes, heights):
            x = pad - box[0]
            if style in {"small_lines", "nutrition"} and self.rng.random() < 0.55:
                # draw faint table separators behind nutrition-like text
                draw.line((0, y + height + 1, layer.width, y + height + 1), fill=(color[0], color[1], color[2], max(10, color[3] // 3)), width=1)
            draw.text((x, y - box[1]), line, font=font, fill=color)
            y += height + line_gap

        if style == "letters_cloud" and self.rng.random() < 0.35:
            alpha = layer.getchannel("A").filter(ImageFilter.GaussianBlur(radius=self.rng.uniform(0.2, 0.8)))
            layer.putalpha(alpha)
        return layer

    def _draw_background_barcode(self, layer: Image.Image) -> None:
        draw = ImageDraw.Draw(layer)
        x = self.rng.randint(-self.image_size // 8, self.image_size - max(20, self.image_size // 3))
        y = self.rng.randint(0, self.image_size - max(20, self.image_size // 7))
        h = self.rng.randint(max(14, self.image_size // 16), max(26, self.image_size // 5))
        bars = self.rng.randint(12, 34)
        color = self._background_text_color(self.rng.randint(24, 95))
        for _ in range(bars):
            w = self.rng.choice([1, 1, 2, 3])
            draw.rectangle((x, y, x + w, y + h + self.rng.randint(-4, 4)), fill=color)
            x += w + self.rng.randint(1, 4)

    def _make_text(self) -> str:
        year = self.rng.randint(2024, 2032)
        yy = year % 100
        month = self.rng.randint(1, 12)
        day = self.rng.randint(1, 28)
        sep = self.rng.choice([".", "/", "-", ":"])
        date_full = f"{day:02d}{sep}{month:02d}{sep}{year}"
        date_short = f"{day:02d}{sep}{month:02d}{sep}{yy:02d}"
        month_year = f"{month:02d}{sep}{year}"
        prefix = self.rng.choice(["", "годен до ", "годен до:", "exp ", "exp:"])
        text = self.rng.choice([
            date_full,
            date_short,
            month_year,
            f"{prefix}{date_full}",
            f"{prefix}{date_short}",
            f"{prefix}{month_year}",
            f"{day:02d}.{month:02d}.{yy:02d}",
            f"годен до {day:02d}/{month:02d}/{yy:02d}",
            f"exp {month:02d}-{yy:02d}",
        ])

        # Occasionally imitate extra machine-printed separators or a compact stamp.
        if self.rng.random() < 0.18:
            text = text.replace(" ", self.rng.choice([" ", "", ":"]))
        if self.rng.random() < 0.12:
            text = f"{text},{self.rng.randint(1, 99):02d}"

        return _filter_alphabet(text.lower())

    def _render_text_layer(self, text: str) -> Image.Image:
        font_size = self.rng.randint(max(10, self.image_size // 18), max(16, self.image_size // 5))
        font = self._choose_font(font_size)
        stroke_width = self.rng.choice([0, 0, 0, 1, 1, 2])
        color = self.rng.choice([
            (10, 10, 10, self.rng.randint(185, 255)),
            (245, 245, 245, self.rng.randint(170, 245)),
            (40, 40, 80, self.rng.randint(170, 240)),
            (85, 25, 20, self.rng.randint(160, 230)),
            (20, 65, 45, self.rng.randint(170, 235)),
        ])
        style = self.rng.choice(["normal", "normal", "bold", "thin", "dotted", "faded"])

        dummy = Image.new("L", (8, 8))
        draw = ImageDraw.Draw(dummy)
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
        text_w = max(1, bbox[2] - bbox[0])
        text_h = max(1, bbox[3] - bbox[1])
        pad = max(8, font_size // 2)
        layer = Image.new("RGBA", (text_w + pad * 2, text_h + pad * 2), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        pos = (pad - bbox[0], pad - bbox[1])

        if style == "dotted":
            mask = Image.new("L", layer.size, 0)
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.text(pos, text, font=font, fill=255, stroke_width=max(0, stroke_width - 1), stroke_fill=255)
            dot_layer = Image.new("RGBA", layer.size, (0, 0, 0, 0))
            dot_draw = ImageDraw.Draw(dot_layer)
            step = self.rng.randint(max(2, font_size // 11), max(3, font_size // 6))
            radius = max(1, step // self.rng.choice([2, 3]))
            for y in range(0, layer.height, step):
                for x in range(0, layer.width, step):
                    jx = x + self.rng.randint(-1, 1)
                    jy = y + self.rng.randint(-1, 1)
                    if 0 <= jx < layer.width and 0 <= jy < layer.height and mask.getpixel((jx, jy)) > 0:
                        dot_draw.ellipse((jx - radius, jy - radius, jx + radius, jy + radius), fill=color)
            layer = dot_layer
        else:
            if style == "bold":
                stroke_width = max(stroke_width, 1)
            if style == "thin":
                color = (color[0], color[1], color[2], max(95, color[3] - 70))
            if style == "faded":
                color = (color[0], color[1], color[2], max(75, color[3] - self.rng.randint(55, 110)))
            draw.text(
                pos,
                text,
                font=font,
                fill=color,
                stroke_width=stroke_width,
                stroke_fill=(color[0], color[1], color[2], min(255, color[3] + 20)),
            )

        if self.rng.random() < 0.25:
            alpha = layer.getchannel("A").filter(ImageFilter.GaussianBlur(radius=self.rng.uniform(0.15, 0.6)))
            layer.putalpha(alpha)
        return layer

    def _distort_text_layer(self, layer: Image.Image) -> Image.Image:
        shear = self.rng.uniform(-0.25, 0.25)
        if abs(shear) > 0.04:
            w, h = layer.size
            new_w = int(w + abs(shear) * h) + 4
            xshift = abs(shear) * h if shear < 0 else 0
            layer = layer.transform(
                (new_w, h),
                Image.Transform.AFFINE,
                (1, shear, -xshift, 0, 1, 0),
                resample=Image.Resampling.BICUBIC,
            )

        angle = self.rng.uniform(-22, 22)
        layer = layer.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)

        if self.rng.random() < 0.25:
            layer = _mild_perspective(layer, self.rng)

        return layer

    def _postprocess(self, image: Image.Image) -> Image.Image:
        if self.rng.random() < 0.20:
            image = image.filter(ImageFilter.GaussianBlur(radius=self.rng.uniform(0.2, 0.9)))
        if self.rng.random() < 0.18:
            image = image.filter(ImageFilter.UnsharpMask(radius=1.0, percent=self.rng.randint(70, 140), threshold=3))
        if self.rng.random() < 0.25:
            arr = np.asarray(image).astype(np.int16)
            noise = np.random.default_rng(self.rng.randint(0, 2**31 - 1)).normal(0, self.rng.uniform(2, 10), arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(arr, mode="RGB")
        return image

    def _choose_font(self, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        if self.fonts:
            font_path = self.rng.choice(self.fonts)
            try:
                return ImageFont.truetype(str(font_path), size=size)
            except Exception:
                pass
        return ImageFont.load_default()


def _read_dataset_yaml(path: Path) -> dict[str, Any]:
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    else:
        data = {}
    data.setdefault("path", str(path.parent.resolve()))
    data.setdefault("train", "images")
    data.setdefault("val", data.get("train", "images"))
    data.setdefault("nc", 1)
    data.setdefault("names", ["expiry_date"])
    return data


def _write_dataset_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _count_train_images(dataset_dir: Path, data: dict[str, Any]) -> int:
    entries = _as_list(data.get("train", "images"))
    total = 0
    for entry in entries:
        if str(entry).replace("\\", "/") == SYNTHETIC_TRAIN_IMAGES:
            continue
        total += _count_images_in_entry(dataset_dir, str(entry))
    return total


def _count_images_in_entry(dataset_dir: Path, entry: str) -> int:
    p = Path(entry)
    if not p.is_absolute():
        p = dataset_dir / p
    if p.is_file() and p.suffix.lower() == ".txt":
        total = 0
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                total += _count_images_in_entry(dataset_dir, line)
        return total
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
        return 1
    if p.is_dir():
        return sum(1 for x in p.rglob("*") if x.is_file() and x.suffix.lower() in IMAGE_EXTENSIONS)
    return 0


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _ensure_synthetic_train_entry(data: dict[str, Any]) -> None:
    entries = [str(x).replace("\\", "/") for x in _as_list(data.get("train", "images"))]
    entries = [x for x in entries if x != SYNTHETIC_TRAIN_IMAGES]
    entries.append(SYNTHETIC_TRAIN_IMAGES)
    data["train"] = entries


def _remove_synthetic_entry_if_present(data: dict[str, Any]) -> None:
    current = data.get("train", "images")
    entries = [str(x).replace("\\", "/") for x in _as_list(current)]
    entries = [x for x in entries if x != SYNTHETIC_TRAIN_IMAGES]
    if len(entries) == 1:
        data["train"] = entries[0]
    else:
        data["train"] = entries


def _normalize_share(value: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = 1.0 / 3.0
    return min(0.8, max(0.01, value))


def _seed_to_int(seed: int | str | None) -> int:
    if seed is None:
        return random.SystemRandom().randint(1, 2**31 - 1)
    if isinstance(seed, int):
        return seed
    result = 0
    for ch in str(seed):
        result = (result * 131 + ord(ch)) % (2**31 - 1)
    return result or 1


def _write_meta(dataset_dir: Path, real_count: int, synth_count: int, target_share: float, imgsz: int) -> None:
    meta = {
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "realTrainImages": real_count,
        "syntheticTrainImages": synth_count,
        "targetSyntheticShare": target_share,
        "actualSyntheticShare": 0 if real_count + synth_count == 0 else synth_count / (real_count + synth_count),
        "imageSize": imgsz,
        "alphabet": ALLOWED_ALPHABET,
        "trainImagesPath": SYNTHETIC_TRAIN_IMAGES,
        "trainLabelsPath": SYNTHETIC_TRAIN_LABELS,
        "backgroundTextEnabled": True,
        "backgroundTextWords": BACKGROUND_WORDS,
    }
    (dataset_dir / "synthetic_dates_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _bbox_to_yolo_line(bbox: tuple[int, int, int, int], width: int, height: int) -> str:
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}\n"


def _load_fonts() -> list[Path]:
    roots = [
        Path("/usr/share/fonts"),
        Path("/usr/local/share/fonts"),
        Path.home() / ".fonts",
        Path("C:/Windows/Fonts"),
        Path("/System/Library/Fonts"),
        Path("/Library/Fonts"),
    ]
    fonts: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for ext in ("*.ttf", "*.otf", "*.ttc"):
            fonts.extend(root.rglob(ext))
    preferred = []
    fallback = []
    for f in fonts:
        name = f.name.lower()
        if any(x in name for x in ("dejavu", "arial", "liberation", "noto", "roboto", "ubuntu", "segoe", "calibri", "times", "cour")):
            preferred.append(f)
        else:
            fallback.append(f)
    return preferred[:80] + fallback[:80]


def _random_packaging_color(rng: random.Random) -> tuple[int, int, int]:
    palettes = [
        (245, 242, 230), (230, 230, 225), (250, 250, 250), (225, 235, 245),
        (235, 225, 210), (210, 225, 210), (245, 230, 235), (235, 235, 245),
        (40, 45, 52), (60, 55, 50), (190, 210, 235), (235, 210, 180),
    ]
    base = np.array(rng.choice(palettes), dtype=np.int16)
    jitter = np.array([rng.randint(-18, 18), rng.randint(-18, 18), rng.randint(-18, 18)], dtype=np.int16)
    c = np.clip(base + jitter, 0, 255)
    return int(c[0]), int(c[1]), int(c[2])


def _near_color(color: Iterable[float], rng: random.Random, *, distance: int = 45) -> tuple[int, int, int]:
    base = np.array(list(color), dtype=np.int16)
    delta = np.array([rng.randint(-distance, distance), rng.randint(-distance, distance), rng.randint(-distance, distance)], dtype=np.int16)
    c = np.clip(base + delta, 0, 255)
    return int(c[0]), int(c[1]), int(c[2])


def _filter_alphabet(text: str) -> str:
    allowed = set(ALLOWED_ALPHABET)
    return "".join(ch for ch in text if ch in allowed).strip() or "годен до 12.06.26"


def _alpha_bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    bbox = image.getchannel("A").getbbox()
    if bbox is None:
        return None
    return int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])


def _clip_bbox(bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)


def _bbox_is_reasonable(bbox: tuple[int, int, int, int], width: int, height: int) -> bool:
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    if bw < max(8, width * 0.08) or bh < max(5, height * 0.025):
        return False
    if bw > width * 0.98 or bh > height * 0.55:
        return False
    return True


def _mild_perspective(layer: Image.Image, rng: random.Random) -> Image.Image:
    w, h = layer.size
    if w < 10 or h < 10:
        return layer
    dx = max(1, int(w * rng.uniform(0.01, 0.06)))
    dy = max(1, int(h * rng.uniform(0.01, 0.10)))
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [
        (rng.randint(0, dx), rng.randint(0, dy)),
        (w - rng.randint(0, dx), rng.randint(0, dy)),
        (w - rng.randint(0, dx), h - rng.randint(0, dy)),
        (rng.randint(0, dx), h - rng.randint(0, dy)),
    ]
    coeffs = _find_perspective_coeffs(dst, src)
    return layer.transform(layer.size, Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)


def _find_perspective_coeffs(pa: list[tuple[int, int]], pb: list[tuple[int, int]]) -> list[float]:
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])
    a = np.array(matrix, dtype=np.float64)
    b = np.array(pb).reshape(8).astype(np.float64)
    res = np.linalg.lstsq(a, b, rcond=None)[0]
    return res.tolist()
