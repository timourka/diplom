from __future__ import annotations

import json
import os
import shutil
import socket
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import requests
import torch
import yaml
from ultralytics import YOLO

APP_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.getenv("TRAINING_CLIENT_DATA", APP_ROOT / "data"))
JOBS_ROOT = DATA_ROOT / "jobs"
BACKEND_URL = os.getenv("PRODUCTS_DATE_BACKEND_URL", os.getenv("BACKEND_URL", "http://127.0.0.1:5099/")).rstrip("/")
API_KEY = os.getenv("TRAINING_CLIENT_API_KEY", os.getenv("TRAINING_SERVICE_API_KEY", ""))
CLIENT_ID = os.getenv("TRAINING_CLIENT_ID", socket.gethostname())
POLL_SECONDS = int(os.getenv("TRAINING_CLIENT_POLL_SECONDS", "15"))
REQUEST_TIMEOUT = int(os.getenv("TRAINING_CLIENT_REQUEST_TIMEOUT", "120"))

JOBS_ROOT.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class TrainConfig:
    base_model: str = "yolov8n.pt"
    epochs: int = 50
    imgsz: int = 640
    batch: int = 16
    device: str = "auto"
    export_int8: bool = True
    export_nms: bool = True
    mobile_format: str = "tflite"
    quantization_fraction: float = 0.3
    workers: int = 0

    @staticmethod
    def from_job(job: dict[str, Any]) -> "TrainConfig":
        return TrainConfig(
            base_model=job.get("baseModel") or "yolov8n.pt",
            epochs=int(job.get("epochs") or 50),
            imgsz=int(job.get("imgSize") or job.get("imgsz") or 640),
            batch=int(job.get("batch") or 16),
            device=job.get("device") or "auto",
            export_int8=bool(job.get("exportInt8", True)),
            export_nms=bool(job.get("exportNms", True)),
            mobile_format=job.get("mobileFormat") or "tflite",
            quantization_fraction=float(job.get("quantizationFraction") or 0.3),
            workers=int(job.get("workers") or os.getenv("YOLO_WORKERS", "0")),
        )


def main() -> None:
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "worker"
    print(f"Training client started. backend={BACKEND_URL}, client_id={CLIENT_ID}, mode={mode}")

    if mode in {"once", "single"}:
        processed = process_next_job()
        print("Job processed." if processed else "No queued jobs.")
        return

    if mode not in {"worker", "loop"}:
        raise SystemExit("Usage: python app.py [worker|once]")

    while True:
        try:
            process_next_job()
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f"Worker loop error: {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc()
        time.sleep(POLL_SECONDS)


def process_next_job() -> bool:
    session = make_session()
    response = session.get(
        f"{BACKEND_URL}/api/training-client/jobs/next",
        params={"clientId": CLIENT_ID},
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code == 204:
        return False

    response.raise_for_status()
    job = response.json()
    job_id = job["jobId"]
    work_dir = JOBS_ROOT / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        update_status(session, job_id, "running", "Python-клиент начал обработку задачи.")
        run_training_job(session, job, work_dir)
        return True
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        print(message, file=sys.stderr)
        try:
            update_status(session, job_id, "failed", message)
        except Exception:
            traceback.print_exc()
        return True


def run_training_job(session: requests.Session, job: dict[str, Any], work_dir: Path) -> None:
    job_id = job["jobId"]
    config = TrainConfig.from_job(job)

    ensure_not_canceled(session, job_id)
    dataset_zip = work_dir / "dataset.zip"
    dataset_dir = work_dir / "dataset"
    runs_dir = work_dir / "runs"
    artifacts_dir = work_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    update_status(session, job_id, "running", "Скачиваю датасет с бэка.")
    download_file(session, f"{BACKEND_URL}/api/training-client/jobs/{job_id}/dataset", dataset_zip)

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(dataset_zip, "r") as zf:
        zf.extractall(dataset_dir)

    normalize_dataset_layout(dataset_dir)
    images_count = count_images(dataset_dir / "images")
    if images_count == 0:
        raise RuntimeError("Dataset must contain images/labels with at least 1 image")

    normalize_dataset_yaml(dataset_dir)
    data_yaml = dataset_dir / "dataset.yaml"

    ensure_not_canceled(session, job_id)
    resolved_device = resolve_train_device(config.device)
    update_status(
        session,
        job_id,
        "running",
        f"Обучение запущено. requested device={config.device}, effective device={resolved_device or 'ultralytics-auto'}, images={images_count}, workers={config.workers}.",
    )

    model = YOLO(config.base_model)
    train_kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": config.epochs,
        "imgsz": config.imgsz,
        "batch": config.batch,
        "project": str(runs_dir),
        "name": "expiry_all",
        "exist_ok": True,
        # On Windows/CUDA, Ultralytics + PyTorch may crash in the DataLoader pin-memory
        # thread with: CUDA error: resource already mapped. Using 0 workers avoids the
        # extra pin-memory thread and is more stable for the local training client.
        "workers": config.workers,
    }
    if resolved_device is not None:
        train_kwargs["device"] = resolved_device

    train_result = model.train(**train_kwargs)

    ensure_not_canceled(session, job_id)
    save_dir = Path(getattr(train_result, "save_dir", runs_dir / "expiry_all"))
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise RuntimeError(f"best.pt was not created: {best_pt}")

    best_artifact = artifacts_dir / "best.pt"
    shutil.copy2(best_pt, best_artifact)

    trained = YOLO(str(best_pt))
    val_kwargs: dict[str, Any] = {"data": str(data_yaml), "imgsz": config.imgsz, "batch": 1, "workers": config.workers}
    if resolved_device is not None:
        val_kwargs["device"] = resolved_device
    metrics = trained.val(**val_kwargs)
    metrics_json = json.dumps(extract_metrics(metrics), ensure_ascii=False)
    update_status(session, job_id, "running", "Обучение завершено, экспортирую мобильную модель.", metrics_json)

    export_kwargs: dict[str, Any] = {
        "format": config.mobile_format,
        "imgsz": config.imgsz,
        "nms": config.export_nms,
    }
    if resolved_device is not None:
        export_kwargs["device"] = resolved_device

    if config.mobile_format == "tflite":
        export_kwargs["int8"] = config.export_int8
        if config.export_int8:
            export_kwargs["data"] = str(data_yaml)
            export_kwargs["fraction"] = config.quantization_fraction

    exported = trained.export(**export_kwargs)
    mobile_artifact = persist_mobile_artifact(exported, config.mobile_format, artifacts_dir)

    ensure_not_canceled(session, job_id)
    upload_artifacts(
        session,
        job_id=job_id,
        best_artifact=best_artifact,
        mobile_artifact=mobile_artifact,
        mobile_format=config.mobile_format,
        metrics_json=metrics_json,
    )


def make_session() -> requests.Session:
    session = requests.Session()
    if API_KEY:
        session.headers.update({"X-Training-Client-Key": API_KEY})
    return session


def update_status(
    session: requests.Session,
    job_id: str,
    status: str,
    message: str,
    metrics_json: str | None = None,
) -> dict[str, Any]:
    payload = {"status": status, "message": message, "metricsJson": metrics_json}
    response = session.post(
        f"{BACKEND_URL}/api/training-client/jobs/{job_id}/status",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def ensure_not_canceled(session: requests.Session, job_id: str) -> None:
    response = session.get(f"{BACKEND_URL}/api/training-client/jobs/{job_id}", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    state = response.json()
    if state.get("cancellationRequested") is True or state.get("status") == "canceled":
        update_status(session, job_id, "canceled", "Python-клиент остановил задачу по запросу администратора.")
        raise RuntimeError("Training job was canceled by administrator")


def download_file(session: requests.Session, url: str, destination: Path) -> None:
    with session.get(url, stream=True, timeout=REQUEST_TIMEOUT) as response:
        response.raise_for_status()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def upload_artifacts(
    session: requests.Session,
    job_id: str,
    best_artifact: Path,
    mobile_artifact: Path,
    mobile_format: str,
    metrics_json: str,
) -> None:
    with best_artifact.open("rb") as best_stream, mobile_artifact.open("rb") as mobile_stream:
        files = {
            "bestWeights": (best_artifact.name, best_stream, "application/octet-stream"),
            "mobileModel": (mobile_artifact.name, mobile_stream, "application/octet-stream"),
        }
        data = {"metricsJson": metrics_json, "mobileFormat": mobile_format}
        response = session.post(
            f"{BACKEND_URL}/api/training-client/jobs/{job_id}/artifacts",
            data=data,
            files=files,
            timeout=max(REQUEST_TIMEOUT, 600),
        )
        response.raise_for_status()


def resolve_train_device(requested: str | None) -> str | None:
    requested = (requested or "auto").strip().lower()

    if requested in {"", "auto"}:
        return None

    if requested == "cpu":
        return "cpu"

    if requested.isdigit():
        index = int(requested)
        if torch.cuda.is_available() and index < torch.cuda.device_count():
            return requested
        raise ValueError(
            f"Requested CUDA device={requested}, but torch.cuda.is_available()={torch.cuda.is_available()} "
            f"and torch.cuda.device_count()={torch.cuda.device_count()} (python={sys.executable})"
        )

    return requested


def normalize_dataset_layout(dataset_dir: Path) -> None:
    if (dataset_dir / "images").exists() and (dataset_dir / "labels").exists():
        return

    nested_images = next((p for p in dataset_dir.rglob("images") if p.is_dir()), None)
    nested_labels = next((p for p in dataset_dir.rglob("labels") if p.is_dir()), None)

    if nested_images and nested_labels:
        target_images = dataset_dir / "images"
        target_labels = dataset_dir / "labels"
        if not target_images.exists():
            shutil.copytree(nested_images, target_images)
        if not target_labels.exists():
            shutil.copytree(nested_labels, target_labels)


def normalize_dataset_yaml(dataset_dir: Path) -> None:
    yaml_path = dataset_dir / "dataset.yaml"

    if yaml_path.exists():
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    else:
        data = {}

    data["path"] = str(dataset_dir.resolve())
    data.setdefault("train", "images")
    data.setdefault("val", data.get("train", "images"))
    data.setdefault("nc", 1)
    data.setdefault("names", ["expiry_date"])

    yaml_path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def count_images(images_dir: Path) -> int:
    if not images_dir.exists():
        return 0
    return sum(1 for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})


def extract_metrics(metrics: Any) -> dict[str, float]:
    result: dict[str, float] = {}
    box = getattr(metrics, "box", None)
    mapping = {
        "map": "mAP50_95",
        "map50": "mAP50",
        "map75": "mAP75",
        "mp": "precision",
        "mr": "recall",
    }

    source = box if box is not None else metrics
    for attr, name in mapping.items():
        value = getattr(source, attr, None)
        if value is not None:
            try:
                result[name] = float(value)
            except (TypeError, ValueError):
                pass

    fitness = getattr(metrics, "fitness", None)
    if fitness is not None:
        try:
            result["fitness"] = float(fitness)
        except (TypeError, ValueError):
            pass

    result["syncedAtUtc"] = datetime.now(timezone.utc).timestamp()
    return result


def persist_mobile_artifact(exported: Any, mobile_format: str, artifacts_dir: Path) -> Path:
    exported_path = Path(str(exported))

    if exported_path.is_file():
        destination = artifacts_dir / f"mobile{exported_path.suffix or '.bin'}"
        shutil.copy2(exported_path, destination)
        return destination

    if exported_path.is_dir():
        destination = artifacts_dir / f"mobile_{mobile_format}.zip"
        shutil.make_archive(str(destination.with_suffix("")), "zip", exported_path)
        return destination

    if mobile_format == "tflite":
        raise RuntimeError(f"Unexpected export result: {exported}")

    raise RuntimeError(f"Unsupported export result for format {mobile_format}: {exported}")


if __name__ == "__main__":
    main()
