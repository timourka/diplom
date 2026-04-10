from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import psutil
import torch
import yaml
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from ultralytics import YOLO

APP_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.getenv("TRAINING_SERVICE_DATA", APP_ROOT / "data"))
JOBS_ROOT = DATA_ROOT / "jobs"
API_KEY = os.getenv("TRAINING_SERVICE_API_KEY", "")
JOBS_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ProductsDate Training Service")


class JobState(BaseModel):
    jobId: str
    status: str
    message: str | None = None
    createdAt: datetime
    startedAt: datetime | None = None
    finishedAt: datetime | None = None
    imagesCount: int = 0
    baseModel: str | None = None
    bestWeightsPath: str | None = None
    mobileModelPath: str | None = None
    mobileFormat: str | None = None
    metricsJson: str | None = None
    processId: int | None = None
    workDir: str


class TrainConfig(BaseModel):
    # Держим дефолты максимально близко к исходному train_expiry_all.py
    baseModel: str = "yolov8n.pt"
    epochs: int = 50
    imgsz: int = 640
    batch: int = 16
    device: str = "auto"
    exportInt8: bool = True
    exportNms: bool = True
    mobileFormat: str = "tflite"
    quantizationFraction: float = 0.3


class JobSummary(BaseModel):
    jobId: str
    status: str
    message: str | None = None
    createdAt: datetime
    startedAt: datetime | None = None
    finishedAt: datetime | None = None
    imagesCount: int = 0
    baseModel: str | None = None
    mobileFormat: str | None = None
    metricsJson: str | None = None


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid X-Api-Key")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "cwd": str(Path.cwd()),
        "pythonExecutable": sys.executable,
        "torchVersion": getattr(torch, "__version__", None),
        "torchCudaVersion": getattr(torch.version, "cuda", None),
        "cudaAvailable": torch.cuda.is_available(),
        "cudaDeviceCount": torch.cuda.device_count(),
    }


@app.get("/jobs", dependencies=[Depends(require_api_key)])
def list_jobs() -> list[dict[str, Any]]:
    jobs: list[JobSummary] = []

    for job_dir in JOBS_ROOT.iterdir():
        if not job_dir.is_dir():
            continue
        state = load_state(job_dir.name)
        if state is None:
            continue
        jobs.append(JobSummary(
            jobId=state.jobId,
            status=state.status,
            message=state.message,
            createdAt=state.createdAt,
            startedAt=state.startedAt,
            finishedAt=state.finishedAt,
            imagesCount=state.imagesCount,
            baseModel=state.baseModel,
            mobileFormat=state.mobileFormat,
            metricsJson=state.metricsJson,
        ))

    jobs.sort(key=lambda x: x.createdAt, reverse=True)
    return [job.model_dump() for job in jobs]


@app.post("/jobs/train", dependencies=[Depends(require_api_key)])
async def start_train_job(
    datasetZip: UploadFile = File(...),
    baseModel: str = Form("yolov8n.pt"),
    epochs: int = Form(50),
    imgsz: int = Form(640),
    batch: int = Form(16),
    device: str = Form("auto"),
    exportInt8: bool = Form(True),
    exportNms: bool = Form(True),
    mobileFormat: str = Form("tflite"),
    quantizationFraction: float = Form(0.3),
) -> dict[str, Any]:
    job_id = uuid.uuid4().hex
    work_dir = JOBS_ROOT / job_id
    uploads_dir = work_dir / "uploads"
    dataset_dir = work_dir / "dataset"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    zip_path = uploads_dir / "dataset.zip"
    with zip_path.open("wb") as f:
        while chunk := await datasetZip.read(1024 * 1024):
            f.write(chunk)

    with ZipFile(zip_path, "r") as zf:
        zf.extractall(dataset_dir)

    normalize_dataset_layout(dataset_dir)
    images_count = count_images(dataset_dir / "images")
    if images_count == 0:
        raise HTTPException(status_code=400, detail="Dataset must contain images/labels with at least 1 image")

    normalize_dataset_yaml(dataset_dir)

    config = TrainConfig(
        baseModel=baseModel,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        exportInt8=exportInt8,
        exportNms=exportNms,
        mobileFormat=mobileFormat,
        quantizationFraction=quantizationFraction,
    )

    state = JobState(
        jobId=job_id,
        status="queued",
        message="Job created",
        createdAt=now_utc(),
        imagesCount=images_count,
        baseModel=baseModel,
        mobileFormat=mobileFormat,
        workDir=str(work_dir),
    )
    save_state(state)
    save_config(job_id, config)

    process = subprocess.Popen(
        [sys.executable, str(APP_ROOT / "job_runner.py"), job_id],
        cwd=str(APP_ROOT),
    )
    state.processId = process.pid
    state.message = f"Job created and queued in separate process pid={process.pid}"
    save_state(state)

    return {
        "jobId": job_id,
        "status": state.status,
        "imagesCount": images_count,
        "message": "Dataset accepted by training-service",
    }


@app.get("/jobs/{job_id}", dependencies=[Depends(require_api_key)])
def get_job(job_id: str) -> dict[str, Any]:
    state = load_state(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return state.model_dump()


@app.post("/jobs/{job_id}/cancel", dependencies=[Depends(require_api_key)])
def cancel_job(job_id: str) -> dict[str, Any]:
    state = load_state(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if state.status in {"completed", "failed", "canceled"}:
        return state.model_dump()

    process_stopped = False
    if state.processId:
        process_stopped = terminate_process_tree(state.processId)

    state.status = "canceled"
    state.finishedAt = now_utc()
    state.message = (
        f"Training job canceled by request. Process pid={state.processId} was stopped."
        if process_stopped
        else "Training job canceled by request. Process was already not running."
    )
    state.processId = None
    save_state(state)
    return state.model_dump()


@app.get("/jobs/{job_id}/artifacts/{artifact}", dependencies=[Depends(require_api_key)])
def download_artifact(job_id: str, artifact: str):
    state = load_state(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Job not found")

    rel_path = {
        "best": state.bestWeightsPath,
        "mobile": state.mobileModelPath,
    }.get(artifact)

    if not rel_path:
        raise HTTPException(status_code=404, detail="Artifact not available")

    file_path = Path(state.workDir) / rel_path
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Artifact file missing")

    media_type = "application/octet-stream"
    if file_path.suffix == ".pt":
        media_type = "application/x-pytorch"
    elif file_path.suffix == ".zip":
        media_type = "application/zip"

    return FileResponse(file_path, media_type=media_type, filename=file_path.name)


def run_job(job_id: str) -> None:
    state = load_state(job_id)
    if state is None:
        return

    if state.status == "canceled":
        return

    config = load_config(job_id)
    if config is None:
        state.status = "failed"
        state.message = "Config file is missing"
        state.finishedAt = now_utc()
        save_state(state)
        return

    try:
        state.status = "running"
        state.processId = os.getpid()
        resolved_device = resolve_train_device(config.device)
        state.message = f"Training started with requested device={config.device}, effective device={resolved_device or 'ultralytics-auto'}"
        state.startedAt = now_utc()
        save_state(state)

        work_dir = Path(state.workDir)
        dataset_dir = work_dir / "dataset"
        data_yaml = dataset_dir / "dataset.yaml"
        runs_dir = work_dir / "runs"
        artifacts_dir = work_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Исходный пользовательский скрипт train_expiry_all.py: data, epochs, imgsz, batch.
        # Тут добавляем только project/name/device ради job-oriented пайплайна.
        model = YOLO(config.baseModel)
        train_kwargs = {
            "data": str(data_yaml),
            "epochs": config.epochs,
            "imgsz": config.imgsz,
            "batch": config.batch,
            "project": str(runs_dir),
            "name": "expiry_all",
            "exist_ok": True,
        }
        # Критично: исходный пользовательский train_expiry_all.py не задавал device вообще.
        # Для режима auto не прокидываем device в Ultralytics, чтобы он сам выбрал GPU/CPU
        # ровно так же, как это происходило в рабочем локальном скрипте.
        if resolved_device is not None:
            train_kwargs["device"] = resolved_device
        train_result = model.train(**train_kwargs)

        state = load_state(job_id) or state
        if state.status == "canceled":
            return

        save_dir = Path(getattr(train_result, "save_dir", runs_dir / "expiry_all"))
        best_pt = save_dir / "weights" / "best.pt"
        if not best_pt.exists():
            raise RuntimeError(f"best.pt was not created: {best_pt}")

        best_artifact = artifacts_dir / "best.pt"
        shutil.copy2(best_pt, best_artifact)
        state.bestWeightsPath = str(best_artifact.relative_to(work_dir)).replace("\\", "/")
        save_state(state)

        trained = YOLO(str(best_pt))
        val_kwargs = {"data": str(data_yaml), "imgsz": config.imgsz, "batch": 1}
        if resolved_device is not None:
            val_kwargs["device"] = resolved_device
        metrics = trained.val(**val_kwargs)
        state.metricsJson = json.dumps(extract_metrics(metrics), ensure_ascii=False)
        state.message = "Training finished, exporting mobile model"
        save_state(state)

        export_kwargs: dict[str, Any] = {
            "format": config.mobileFormat,
            "imgsz": config.imgsz,
            "nms": config.exportNms,
        }
        if resolved_device is not None:
            export_kwargs["device"] = resolved_device

        if config.mobileFormat == "tflite":
            export_kwargs["int8"] = config.exportInt8
            if config.exportInt8:
                export_kwargs["data"] = str(data_yaml)
                export_kwargs["fraction"] = config.quantizationFraction

        exported = trained.export(**export_kwargs)
        mobile_artifact = persist_mobile_artifact(exported, config.mobileFormat, artifacts_dir)
        state.mobileModelPath = str(mobile_artifact.relative_to(work_dir)).replace("\\", "/")
        state.status = "completed"
        state.message = f"Training and mobile export completed on device={resolved_device}"
        state.finishedAt = now_utc()
        state.processId = None
        save_state(state)
    except Exception as exc:  # pragma: no cover - defensive path
        state = load_state(job_id) or state
        if state.status == "canceled":
            return
        state.status = "failed"
        state.finishedAt = now_utc()
        state.processId = None
        state.message = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        save_state(state)


def terminate_process_tree(pid: int) -> bool:
    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return False

    children = process.children(recursive=True)
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass

    try:
        process.terminate()
    except psutil.NoSuchProcess:
        return False

    gone, alive = psutil.wait_procs([*children, process], timeout=5)
    for proc in alive:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass

    return True


def resolve_train_device(requested: str | None) -> str | None:
    requested = (requested or "auto").strip().lower()

    # Максимально повторяем исходный train_expiry_all.py:
    # если device не задан явно, вообще не передаем его в Ultralytics.
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

    # Для строк вроде '0,1' или 'cuda:0' передаем как есть: пользователь явно попросил конкретное устройство.
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

    # На Windows Ultralytics может резолвить относительный path не от файла yaml, а от cwd процесса.
    # Поэтому фиксируем абсолютный путь до job dataset directory.
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


def state_path(job_id: str) -> Path:
    return JOBS_ROOT / job_id / "job_state.json"


def config_path(job_id: str) -> Path:
    return JOBS_ROOT / job_id / "job_config.json"


def save_state(state: JobState) -> None:
    state_path(state.jobId).write_text(state.model_dump_json(indent=2), encoding="utf-8")


def load_state(job_id: str) -> JobState | None:
    path = state_path(job_id)
    if not path.exists():
        return None
    return JobState.model_validate_json(path.read_text(encoding="utf-8"))


def save_config(job_id: str, config: TrainConfig) -> None:
    config_path(job_id).write_text(config.model_dump_json(indent=2), encoding="utf-8")


def load_config(job_id: str) -> TrainConfig | None:
    path = config_path(job_id)
    if not path.exists():
        return None
    return TrainConfig.model_validate_json(path.read_text(encoding="utf-8"))
