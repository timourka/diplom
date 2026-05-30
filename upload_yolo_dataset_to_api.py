"""
Одноразовый импорт YOLO-датасета в ProductsDateAPI как ErrorReport.

Что делает:
1) Берёт локальный YOLO-датасет.
2) Собирает zip со структурой:
      images/frame_00001.jpg
      labels/frame_00001.txt
3) POST /api/Auth/login
4) POST /api/error-reports/upload-dataset  (multipart: datasetZip, comment)
5) PUT  /api/admin/error-reports/{reportId}/approve  (опционально)
6) POST /api/admin/training/start  (опционально)

Зависимости:
    pip install requests pillow

Пример:
    python upload_yolo_dataset_to_api.py \
      --dataset ./datasets/expdate_yolo \
      --email admin@example.com \
      --password password \
      --limit 200 \
      --approve \
      --start-training
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Pair:
    image_path: Path
    label_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://111.88.146.2:5099")
    parser.add_argument("--dataset", required=True, help="Путь к YOLO-датасету")
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--comment", default="Импорт внешнего YOLO-датасета для дообучения детектора области даты")
    parser.add_argument("--limit", type=int, default=None, help="Сколько пар image/label загрузить. Для теста поставь 5-20")
    parser.add_argument("--offset", type=int, default=0, help="С какой пары начать. Удобно для загрузки батчами")
    parser.add_argument("--approve", action="store_true", help="Сразу пометить ErrorReport как Approved")
    parser.add_argument("--start-training", action="store_true", help="Сразу создать задачу обучения")
    parser.add_argument("--base-model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mobile-format", default="tflite")
    parser.add_argument("--export-int8", action="store_true", help="Поставить exportInt8=true")
    parser.add_argument("--no-export-nms", action="store_true", help="Поставить exportNms=false")
    parser.add_argument("--quantization-fraction", type=float, default=0.3)
    parser.add_argument("--keep-zip", default=None, help="Куда сохранить собранный zip, например ./expdate_import.zip")
    parser.add_argument("--dry-run", action="store_true", help="Только собрать zip, не отправлять в API")
    return parser.parse_args()


def first_existing(*paths: Path) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def find_pairs(dataset_root: Path) -> List[Pair]:
    """
    Поддерживает варианты:
      dataset/images/*.jpg + dataset/labels/*.txt
      dataset/images/train/*.jpg + dataset/labels/train/*.txt
      dataset/images/val/*.jpg + dataset/labels/val/*.txt
      произвольные вложенные папки внутри images/ и labels/
    """
    dataset_root = dataset_root.resolve()
    images_root = dataset_root / "images"
    labels_root = dataset_root / "labels"

    if not images_root.exists():
        raise FileNotFoundError(f"Не найдена папка images: {images_root}")
    if not labels_root.exists():
        raise FileNotFoundError(f"Не найдена папка labels: {labels_root}")

    pairs: List[Pair] = []

    for image_path in sorted(images_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
            continue

        rel = image_path.relative_to(images_root)
        label_path = labels_root / rel.with_suffix(".txt")

        # fallback: поиск по stem, если структура labels отличается
        if not label_path.exists():
            candidates = list(labels_root.rglob(image_path.stem + ".txt"))
            if candidates:
                label_path = candidates[0]

        if label_path.exists():
            pairs.append(Pair(image_path=image_path, label_path=label_path))

    if not pairs:
        raise RuntimeError("Не найдено ни одной пары image+label")

    return pairs


def normalize_yolo_label(label_path: Path) -> str:
    """
    Приводит label к одному классу 0.
    Оставляет только первые 5 значений YOLO: class xc yc w h.
    """
    out_lines: List[str] = []
    for raw in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) < 5:
            continue

        try:
            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            continue

        # защита от мусорных bbox
        if w <= 0 or h <= 0:
            continue

        xc = min(max(xc, 0.0), 1.0)
        yc = min(max(yc, 0.0), 1.0)
        w = min(max(w, 0.0), 1.0)
        h = min(max(h, 0.0), 1.0)

        out_lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    return "\n".join(out_lines) + ("\n" if out_lines else "")


def save_as_jpg(src: Path, dst: Path) -> None:
    """
    Сохраняет все изображения как JPG.
    Это удобно для твоего AdminErrorReportsController, где просмотр кадров ожидает frame_00001.jpg.
    """
    with Image.open(src) as img:
        img = img.convert("RGB")
        img.save(dst, format="JPEG", quality=95)


def build_import_zip(pairs: List[Pair], zip_path: Path) -> Tuple[Path, int]:
    work_dir = Path(tempfile.mkdtemp(prefix="api_dataset_import_"))
    images_out = work_dir / "images"
    labels_out = work_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    copied = 0
    for pair in pairs:
        label_text = normalize_yolo_label(pair.label_path)
        if not label_text.strip():
            print(f"[SKIP] пустая/битая разметка: {pair.label_path}")
            continue

        copied += 1
        stem = f"frame_{copied:05d}"
        dst_img = images_out / f"{stem}.jpg"
        dst_lbl = labels_out / f"{stem}.txt"

        save_as_jpg(pair.image_path, dst_img)
        dst_lbl.write_text(label_text, encoding="utf-8")

    if copied == 0:
        raise RuntimeError("После фильтрации не осталось валидных примеров")

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(work_dir.rglob("*")):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(work_dir).as_posix())

    shutil.rmtree(work_dir, ignore_errors=True)
    return zip_path, copied


def login(session: requests.Session, base_url: str, email: str, password: str) -> str:
    url = f"{base_url.rstrip('/')}/api/Auth/login"
    response = session.post(url, json={"email": email, "password": password}, timeout=30)
    if response.status_code >= 400:
        raise RuntimeError(f"Login failed: HTTP {response.status_code}\n{response.text}")

    data = response.json()
    token = data.get("accessToken") or data.get("AccessToken")
    if not token:
        raise RuntimeError(f"В ответе login не найден accessToken: {data}")
    return token


def upload_dataset(session: requests.Session, base_url: str, zip_path: Path, comment: str) -> Dict:
    url = f"{base_url.rstrip('/')}/api/error-reports/upload-dataset"
    with zip_path.open("rb") as f:
        response = session.post(
            url,
            data={"comment": comment},
            files={"datasetZip": (zip_path.name, f, "application/zip")},
            timeout=300,
        )

    if response.status_code >= 400:
        raise RuntimeError(f"Upload failed: HTTP {response.status_code}\n{response.text}")

    return response.json()


def approve_report(session: requests.Session, base_url: str, report_id: int) -> None:
    url = f"{base_url.rstrip('/')}/api/admin/error-reports/{report_id}/approve"
    response = session.put(url, json={"approved": True}, timeout=30)
    if response.status_code >= 400:
        raise RuntimeError(f"Approve failed: HTTP {response.status_code}\n{response.text}")


def start_training(session: requests.Session, base_url: str, args: argparse.Namespace) -> Dict:
    url = f"{base_url.rstrip('/')}/api/admin/training/start"
    payload = {
        "baseModel": args.base_model,
        "epochs": args.epochs,
        "imgSize": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "exportInt8": bool(args.export_int8),
        "exportNms": not bool(args.no_export_nms),
        "mobileFormat": args.mobile_format,
        "quantizationFraction": args.quantization_fraction,
    }
    response = session.post(url, json=payload, timeout=120)
    if response.status_code >= 400:
        raise RuntimeError(f"Start training failed: HTTP {response.status_code}\n{response.text}")
    return response.json()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset)

    pairs = find_pairs(dataset_root)
    print(f"[DATASET] найдено пар image+label: {len(pairs)}")

    if args.offset:
        pairs = pairs[args.offset:]
    if args.limit is not None:
        pairs = pairs[:args.limit]

    print(f"[DATASET] будет упаковано: {len(pairs)}")

    if args.keep_zip:
        zip_path = Path(args.keep_zip)
    else:
        zip_path = Path(tempfile.gettempdir()) / "products_date_dataset_import.zip"

    zip_path, count = build_import_zip(pairs, zip_path)
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"[ZIP] {zip_path} | {count} изображений | {size_mb:.2f} MB")

    if size_mb > 380:
        print("[WARN] zip близок к RequestSizeLimit 400 MB. Лучше грузить меньшими батчами через --limit/--offset")

    if args.dry_run:
        print("[DRY RUN] API-запросы не выполнялись")
        return

    session = requests.Session()
    token = login(session, args.base_url, args.email, args.password)
    session.headers.update({"Authorization": f"Bearer {token}"})
    print("[AUTH] OK")

    upload_result = upload_dataset(session, args.base_url, zip_path, args.comment)
    print("[UPLOAD] OK")
    print(json.dumps(upload_result, ensure_ascii=False, indent=2))

    report_id = upload_result.get("reportId") or upload_result.get("ReportId")
    if not report_id:
        raise RuntimeError(f"Не найден reportId в ответе upload: {upload_result}")

    if args.approve:
        approve_report(session, args.base_url, int(report_id))
        print(f"[APPROVE] reportId={report_id} approved=True")

    if args.start_training:
        if not args.approve:
            print("[WARN] start-training запрошен, но --approve не указан. Неподтверждённый отчёт не попадёт в обучение.")
        training_result = start_training(session, args.base_url, args)
        print("[TRAINING] задача создана")
        print(json.dumps(training_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
