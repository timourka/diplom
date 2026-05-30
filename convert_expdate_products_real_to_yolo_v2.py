# -*- coding: utf-8 -*-
"""
ExpDate Products-Real -> YOLO converter, version 2.

Этот вариант специально поддерживает реальный формат ExpDate:
{
  "img_00001.jpg": {
    "height": 1008,
    "width": 756,
    "ann": [
      {"cls": "date", "bbox": [289, 660, 418, 673], "transcription": "..."}
    ]
  }
}

Ожидаемый вход:
  Products-Real/
    train/
      images/
      annotations.json
    evaluation/
      images/
      annotations.json

Выход:
  datasets/expdate_products_real_yolo/
    images/train/*.jpg
    labels/train/*.txt
    images/val/*.jpg
    labels/val/*.txt
    data.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Папка Products-Real")
    parser.add_argument("--out", default="datasets/expdate_products_real_yolo", help="Куда сохранить YOLO-датасет")
    parser.add_argument(
        "--include-classes",
        default="date,exp",
        help="Какие классы ExpDate брать и сводить в один класс YOLO 0. Например: date,exp,due,prod,code",
    )
    parser.add_argument("--limit", type=int, default=None, help="Ограничить число изображений на split для проверки")
    parser.add_argument("--inspect", action="store_true", help="Показать структуру и статистику без конвертации")
    parser.add_argument("--no-clean", action="store_true", help="Не очищать out перед конвертацией")
    return parser.parse_args()


def find_images(images_dir: Path) -> Dict[str, Path]:
    images: Dict[str, Path] = {}
    if not images_dir.exists():
        return images

    for path in images_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMG_EXTS:
            images[path.name] = path
            images[path.stem] = path

    return images


def find_annotations_json(split_dir: Path) -> Path:
    preferred = split_dir / "annotations.json"
    if preferred.exists():
        return preferred

    candidates = sorted(split_dir.rglob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"Не найден annotations.json в {split_dir}")

    # Обычно нужный файл самый крупный.
    return max(candidates, key=lambda p: p.stat().st_size)


def load_expdate_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError(f"{path} должен быть JSON-объектом, а не {type(data).__name__}")

    return data


def get_image_size_from_file(image_path: Path) -> Tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "В annotations.json нет width/height, а Pillow не установлен. "
            "Установи: pip install pillow"
        ) from exc

    with Image.open(image_path) as image:
        return image.size


def clamp_bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_w: int,
    image_h: int,
) -> Optional[Tuple[float, float, float, float]]:
    x1 = max(0.0, min(float(x1), float(image_w)))
    y1 = max(0.0, min(float(y1), float(image_h)))
    x2 = max(0.0, min(float(x2), float(image_w)))
    y2 = max(0.0, min(float(y2), float(image_h)))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def bbox_to_yolo_line(
    bbox: Iterable[Any],
    image_w: int,
    image_h: int,
) -> Optional[str]:
    values = list(bbox)
    if len(values) != 4:
        return None

    try:
        x1, y1, x2, y2 = [float(v) for v in values]
    except Exception:
        return None

    clamped = clamp_bbox(x1, y1, x2, y2, image_w, image_h)
    if clamped is None:
        return None

    x1, y1, x2, y2 = clamped

    x_center = ((x1 + x2) / 2.0) / image_w
    y_center = ((y1 + y2) / 2.0) / image_h
    width = (x2 - x1) / image_w
    height = (y2 - y1) / image_h

    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def inspect_split(src: Path, split_name: str) -> None:
    split_dir = src / split_name
    images_dir = split_dir / "images"

    images = find_images(images_dir)
    print(f"[{split_name}] images: {len({p for p in images.values()})} in {images_dir}")

    try:
        ann_path = find_annotations_json(split_dir)
    except FileNotFoundError as exc:
        print(f"[{split_name}] {exc}")
        return

    data = load_expdate_json(ann_path)
    ann_count = 0
    class_counts: Dict[str, int] = {}

    for item in data.values():
        if not isinstance(item, dict):
            continue
        for ann in item.get("ann", []):
            if not isinstance(ann, dict):
                continue
            cls = str(ann.get("cls", "")).lower()
            ann_count += 1
            class_counts[cls] = class_counts.get(cls, 0) + 1

    print(f"[{split_name}] annotation file: {ann_path}")
    print(f"[{split_name}] json image keys: {len(data)}")
    print(f"[{split_name}] annotations: {ann_count}")
    print(f"[{split_name}] classes: {class_counts}")


def convert_split(
    src: Path,
    out: Path,
    split_name: str,
    yolo_split_name: str,
    include_classes: set[str],
    limit: Optional[int],
) -> None:
    split_dir = src / split_name
    images_dir = split_dir / "images"

    if not split_dir.exists():
        print(f"[SKIP] нет папки {split_dir}")
        return

    ann_path = find_annotations_json(split_dir)
    data = load_expdate_json(ann_path)
    images = find_images(images_dir)

    out_images = out / "images" / yolo_split_name
    out_labels = out / "labels" / yolo_split_name
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    converted_images = 0
    converted_boxes = 0
    skipped_no_image = 0
    skipped_class = 0
    skipped_bad_bbox = 0
    skipped_empty = 0

    print(f"[{split_name}] annotation: {ann_path}")
    print(f"[{split_name}] json image keys: {len(data)}")

    for image_name, item in data.items():
        if limit is not None and converted_images >= limit:
            break

        if not isinstance(item, dict):
            skipped_empty += 1
            continue

        img_path = images.get(Path(image_name).name) or images.get(Path(image_name).stem)
        if img_path is None:
            skipped_no_image += 1
            continue

        width = item.get("width")
        height = item.get("height")

        if width is None or height is None:
            width, height = get_image_size_from_file(img_path)
        else:
            width = int(width)
            height = int(height)

        yolo_lines: List[str] = []

        for ann in item.get("ann", []):
            if not isinstance(ann, dict):
                skipped_bad_bbox += 1
                continue

            cls = str(ann.get("cls", "")).lower().strip()
            if cls not in include_classes:
                skipped_class += 1
                continue

            line = bbox_to_yolo_line(ann.get("bbox", []), width, height)
            if line is None:
                skipped_bad_bbox += 1
                continue

            yolo_lines.append(line)

        if not yolo_lines:
            skipped_empty += 1
            continue

        dst_image = out_images / img_path.name
        dst_label = out_labels / f"{img_path.stem}.txt"

        shutil.copy2(img_path, dst_image)
        dst_label.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

        converted_images += 1
        converted_boxes += len(yolo_lines)

    print(
        f"[{split_name}] converted images: {converted_images} | "
        f"boxes: {converted_boxes} | "
        f"skipped_no_image: {skipped_no_image} | "
        f"skipped_class: {skipped_class} | "
        f"skipped_bad_bbox: {skipped_bad_bbox} | "
        f"skipped_empty_images: {skipped_empty}"
    )


def write_data_yaml(out: Path) -> None:
    content = f"""path: {out.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: expiry_date
"""
    (out / "data.yaml").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()

    src = Path(args.src)
    out = Path(args.out)
    include_classes = {part.strip().lower() for part in args.include_classes.split(",") if part.strip()}

    if args.inspect:
        print(f"[SRC] {src.resolve()}")
        inspect_split(src, "train")
        inspect_split(src, "evaluation")
        return

    if out.exists() and not args.no_clean:
        print(f"[WARN] очищаю старую папку: {out}")
        shutil.rmtree(out)

    convert_split(src, out, "train", "train", include_classes, args.limit)
    convert_split(src, out, "evaluation", "val", include_classes, args.limit)
    write_data_yaml(out)

    print()
    print(f"[OK] YOLO dataset saved: {out.resolve()}")
    print(f"[OK] YAML: {(out / 'data.yaml').resolve()}")
    print()
    print("Проверка обучения:")
    print(f"  yolo detect train model=yolov8n.pt data={out / 'data.yaml'} epochs=10 imgsz=640")
    print()
    print("Загрузка в твою API:")
    print(f"  python upload_yolo_dataset_to_api.py --dataset {out} --email <email> --password <password> --limit 20 --approve")


if __name__ == "__main__":
    main()
