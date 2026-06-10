"""
Загрузка локального YOLO-датасета datasets/real_photo_ds2 в ProductsDateAPI как ErrorReport.

Ожидаемая структура:
  datasets/real_photo_ds2/images/*.jpg|png|webp...
  datasets/real_photo_ds2/labels/*.txt

Скрипт:
  1) находит пары image+label;
  2) упаковывает их в zip со структурой images/frame_00001.jpg + labels/frame_00001.txt;
  3) логинится в API;
  4) отправляет POST /api/error-reports/upload-dataset;
  5) опционально approve/start-training.

Зависимости:
  pip install requests pillow

Пример:
  python upload_real_photo_ds2_to_api.py --email a --password a

Партиями по 50:
  python upload_real_photo_ds2_to_api.py --email a --password a --batch-size 50
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Pair:
    image_path: Path
    label_path: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload real_photo_ds2 YOLO dataset to ProductsDateAPI")
    p.add_argument("--base-url", default="http://111.88.146.2:5099")
    p.add_argument("--dataset", default=r"D:\diplom\datasets\real_photo_ds2")
    p.add_argument("--email", required=True)
    p.add_argument("--password", required=True)
    p.add_argument("--comment", default="Импорт реальных фотографий с телефона для дообучения детектора области даты")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None, help="Если указано, загружает несколькими отчётами по N изображений")
    p.add_argument("--approve", action="store_true", help="Сразу отметить отчёт Approved")
    p.add_argument("--start-training", action="store_true", help="После загрузки создать задачу обучения")
    p.add_argument("--base-model", default="yolov8n.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="auto")
    p.add_argument("--mobile-format", default="tflite")
    p.add_argument("--export-int8", action="store_true")
    p.add_argument("--no-export-nms", action="store_true")
    p.add_argument("--quantization-fraction", type=float, default=0.3)
    p.add_argument("--upload-timeout", type=int, default=900)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--keep-zips-dir", default=None, help="Папка, куда сохранить собранные zip")
    return p.parse_args()


def find_pairs(dataset_root: Path) -> List[Pair]:
    dataset_root = dataset_root.resolve()
    images_root = dataset_root / "images"
    labels_root = dataset_root / "labels"
    if not images_root.exists():
        raise FileNotFoundError(f"Не найдена папка images: {images_root}")
    if not labels_root.exists():
        raise FileNotFoundError(f"Не найдена папка labels: {labels_root}")

    pairs: List[Pair] = []
    for img_path in sorted(images_root.rglob("*")):
        if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = img_path.relative_to(images_root)
        lbl_path = labels_root / rel.with_suffix(".txt")
        if not lbl_path.exists():
            candidates = list(labels_root.rglob(img_path.stem + ".txt"))
            if candidates:
                lbl_path = candidates[0]
        if lbl_path.exists():
            pairs.append(Pair(img_path, lbl_path))
        else:
            print(f"[SKIP] нет label для {img_path}")

    if not pairs:
        raise RuntimeError("Не найдено ни одной пары image+label")
    return pairs


def normalize_yolo_label(label_path: Path) -> str:
    out: List[str] = []
    for raw in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) < 5:
            continue
        try:
            xc, yc, w, h = map(float, parts[1:5])
        except ValueError:
            continue
        if w <= 0 or h <= 0:
            continue
        xc = min(max(xc, 0.0), 1.0)
        yc = min(max(yc, 0.0), 1.0)
        w = min(max(w, 0.0), 1.0)
        h = min(max(h, 0.0), 1.0)
        out.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return "\n".join(out) + ("\n" if out else "")


def save_as_jpg(src: Path, dst: Path) -> None:
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.save(dst, format="JPEG", quality=95)


def build_zip(pairs: List[Pair], zip_path: Path) -> Tuple[Path, int]:
    tmp = Path(tempfile.mkdtemp(prefix="real_photo_ds2_upload_"))
    images_out = tmp / "images"
    labels_out = tmp / "labels"
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
        save_as_jpg(pair.image_path, images_out / f"{stem}.jpg")
        (labels_out / f"{stem}.txt").write_text(label_text, encoding="utf-8")

    if copied == 0:
        raise RuntimeError("После фильтрации не осталось валидных примеров")

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(tmp.rglob("*")):
            if fp.is_file():
                zf.write(fp, fp.relative_to(tmp).as_posix())
    shutil.rmtree(tmp, ignore_errors=True)
    return zip_path, copied


def login(session: requests.Session, base_url: str, email: str, password: str) -> str:
    r = session.post(f"{base_url.rstrip('/')}/api/Auth/login", json={"email": email, "password": password}, timeout=30)
    if r.status_code >= 400:
        raise RuntimeError(f"Login failed: HTTP {r.status_code}\n{r.text}")
    data = r.json()
    token = data.get("accessToken") or data.get("AccessToken")
    if not token:
        raise RuntimeError(f"В ответе login не найден accessToken: {data}")
    return token


def upload_dataset(session: requests.Session, base_url: str, zip_path: Path, comment: str, timeout: int) -> Dict:
    url = f"{base_url.rstrip('/')}/api/error-reports/upload-dataset"
    print(f"[UPLOAD] {zip_path} -> {url}")
    with zip_path.open("rb") as f:
        r = session.post(
            url,
            data={"comment": comment},
            files={"datasetZip": (zip_path.name, f, "application/zip")},
            timeout=timeout,
        )
    if r.status_code >= 400:
        raise RuntimeError(f"Upload failed: HTTP {r.status_code}\n{r.text}")
    return r.json()


def approve_report(session: requests.Session, base_url: str, report_id: int) -> None:
    r = session.put(f"{base_url.rstrip('/')}/api/admin/error-reports/{report_id}/approve", json={"approved": True}, timeout=30)
    if r.status_code >= 400:
        raise RuntimeError(f"Approve failed: HTTP {r.status_code}\n{r.text}")


def start_training(session: requests.Session, base_url: str, args: argparse.Namespace) -> Dict:
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
    r = session.post(f"{base_url.rstrip('/')}/api/admin/training/start", json=payload, timeout=120)
    if r.status_code >= 400:
        raise RuntimeError(f"Start training failed: HTTP {r.status_code}\n{r.text}")
    return r.json()


def chunked(items: List[Pair], batch_size: Optional[int]) -> List[List[Pair]]:
    if not batch_size:
        return [items]
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset)
    pairs = find_pairs(dataset)
    print(f"[DATASET] найдено пар image+label: {len(pairs)}")

    if args.offset:
        pairs = pairs[args.offset:]
    if args.limit is not None:
        pairs = pairs[:args.limit]
    print(f"[DATASET] будет обработано: {len(pairs)}")

    batches = chunked(pairs, args.batch_size)
    print(f"[DATASET] батчей: {len(batches)}")

    session = requests.Session()
    if not args.dry_run:
        token = login(session, args.base_url, args.email, args.password)
        session.headers.update({"Authorization": f"Bearer {token}"})
        print("[AUTH] OK")

    uploaded_report_ids: List[int] = []
    zips_dir = Path(args.keep_zips_dir) if args.keep_zips_dir else None

    for idx, batch_pairs in enumerate(batches, start=1):
        if zips_dir:
            zip_path = zips_dir / f"real_photo_ds2_import_part_{idx:03d}.zip"
        else:
            zip_path = Path(tempfile.gettempdir()) / f"real_photo_ds2_import_part_{idx:03d}.zip"

        zip_path, count = build_zip(batch_pairs, zip_path)
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"[ZIP {idx}/{len(batches)}] {zip_path} | {count} изображений | {size_mb:.2f} MB")

        if args.dry_run:
            continue

        result = upload_dataset(session, args.base_url, zip_path, f"{args.comment} | batch {idx}/{len(batches)}", args.upload_timeout)
        print("[UPLOAD] OK")
        print(json.dumps(result, ensure_ascii=False, indent=2))

        report_id = result.get("reportId") or result.get("ReportId")
        if not report_id:
            raise RuntimeError(f"Не найден reportId в ответе upload: {result}")
        report_id = int(report_id)
        uploaded_report_ids.append(report_id)

        if args.approve:
            approve_report(session, args.base_url, report_id)
            print(f"[APPROVE] reportId={report_id} approved=True")

    if args.start_training:
        if not args.approve:
            print("[WARN] start-training запрошен, но --approve не указан. Неподтверждённые отчёты не попадут в обучение.")
        result = start_training(session, args.base_url, args)
        print("[TRAINING] задача создана")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    print(f"[DONE] uploaded reports: {uploaded_report_ids}")


if __name__ == "__main__":
    main()
