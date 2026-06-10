#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Интерактивная разметка реальных фотографий по продуктам + отправка текущей пачки на ProductsDateAPI.

Сценарий:
  1) Открывается папка с НЕразмеченными реальными фото, например D:\\diplom\\datasets\\real_photo_ds3.
  2) Ты размечаешь bbox даты на фото.
  3) Скрипт сохраняет полный НЕкропнутый кадр в out/images и YOLO label в out/labels.
  4) Пока продукт один и тот же — продолжаешь размечать.
  5) Когда продукт сменился — нажимаешь "Отправить пачку". Скрипт отправит только накопленные с прошлого upload изображения.

Зависимости:
  pip install pillow requests

Пример запуска:
  python label_upload_product_batches.py --src D:\\diplom\\datasets\\real_photo_ds3 --out D:\\diplom\\datasets\\real_photo_ds3_labeled

Можно не вводить URL/email/password в аргументах, если заданы переменные окружения:
  PRODUCTS_DATE_BACKEND_URL
  PRODUCTS_DATE_BACKEND_EMAIL
  PRODUCTS_DATE_BACKEND_PASSWORD
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests
from PIL import Image, ImageOps, ImageTk
import tkinter as tk
from tkinter import messagebox, simpledialog

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def ordered(self) -> "BBox":
        x1, x2 = sorted((self.x1, self.x2))
        y1, y2 = sorted((self.y1, self.y2))
        return BBox(x1, y1, x2, y2)

    def clamped(self, w: int, h: int) -> "BBox":
        b = self.ordered()
        x1 = max(0, min(w - 1, b.x1))
        x2 = max(0, min(w, b.x2))
        y1 = max(0, min(h - 1, b.y1))
        y2 = max(0, min(h, b.y2))
        return BBox(x1, y1, x2, y2)

    def valid(self, min_size: int = 4) -> bool:
        b = self.ordered()
        return (b.x2 - b.x1) >= min_size and (b.y2 - b.y1) >= min_size

    def normalized(self, w: int, h: int) -> Tuple[float, float, float, float]:
        b = self.clamped(w, h)
        bw = max(1, b.x2 - b.x1)
        bh = max(1, b.y2 - b.y1)
        xc = b.x1 + bw / 2.0
        yc = b.y1 + bh / 2.0
        return xc / w, yc / h, bw / w, bh / h


@dataclass
class SourceMapRow:
    source: str
    yolo_image: str
    yolo_label: str


class ProductBatchLabeler:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.src = Path(args.src).resolve()
        self.out = Path(args.out).resolve()
        self.images_dir = self.out / "images"
        self.labels_dir = self.out / "labels"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        self.sources_csv = self.out / "sources.csv"
        self.current_batch_file = self.out / "current_batch.txt"
        self.batch_log_csv = self.out / "upload_batches.csv"
        self.state_json = self.out / "state.json"
        self.data_yaml = self.out / "data.yaml"
        self._ensure_files()

        self.paths = self._collect_images(self.src)
        if not self.paths:
            raise SystemExit(f"Не найдено изображений в {self.src}")

        self.source_map: Dict[str, SourceMapRow] = self._load_source_map()
        self.current_batch: Set[str] = self._load_current_batch()
        self.next_yolo_index = self._next_yolo_index()

        state = self._load_state()
        self.index = max(0, min(int(state.get("index", args.start_index)), len(self.paths) - 1))
        self.current_product = str(state.get("product", ""))

        self.img: Optional[Image.Image] = None
        self.current_path: Optional[Path] = None
        self.tk_img = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.start_xy: Optional[Tuple[int, int]] = None
        self.current_bbox: Optional[BBox] = None
        self.current_rect_id = None
        self.saved_rect_ids: List[int] = []

        self.root = tk.Tk()
        self.root.title("Разметка дат по продуктам + отправка пачками")
        self.root.geometry("1250x880")
        self._build_ui()
        self._bind_keys()
        self._load_current()

    def _ensure_files(self) -> None:
        if not self.sources_csv.exists():
            with self.sources_csv.open("w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(["source", "yolo_image", "yolo_label"])
        if not self.batch_log_csv.exists():
            with self.batch_log_csv.open("w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(["time", "batch_id", "report_id", "product", "count", "comment"])
        if not self.current_batch_file.exists():
            self.current_batch_file.write_text("", encoding="utf-8")
        self.data_yaml.write_text(
            f"path: {self.out.as_posix()}\n"
            f"train: images\n"
            f"val: images\n"
            f"names:\n"
            f"  0: expiry_date\n",
            encoding="utf-8",
        )

    def _collect_images(self, root: Path) -> List[Path]:
        out_resolved = self.out.resolve()
        result: List[Path] = []
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
                continue
            try:
                # не сканировать выходную папку, если она случайно внутри source
                p.resolve().relative_to(out_resolved)
                continue
            except ValueError:
                pass
            result.append(p)
        return sorted(result)

    def _load_source_map(self) -> Dict[str, SourceMapRow]:
        rows: Dict[str, SourceMapRow] = {}
        with self.sources_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                src = r.get("source") or ""
                img = r.get("yolo_image") or ""
                lbl = r.get("yolo_label") or ""
                if src and img and lbl:
                    rows[src] = SourceMapRow(src, img, lbl)
        return rows

    def _load_current_batch(self) -> Set[str]:
        if not self.current_batch_file.exists():
            return set()
        return {line.strip() for line in self.current_batch_file.read_text(encoding="utf-8").splitlines() if line.strip()}

    def _save_current_batch(self) -> None:
        self.current_batch_file.write_text("\n".join(sorted(self.current_batch)) + ("\n" if self.current_batch else ""), encoding="utf-8")

    def _load_state(self) -> Dict:
        if not self.state_json.exists():
            return {}
        try:
            return json.loads(self.state_json.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_state(self) -> None:
        self.state_json.write_text(
            json.dumps({"index": self.index, "product": self.product_var.get().strip()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _next_yolo_index(self) -> int:
        nums: List[int] = []
        for p in self.images_dir.glob("*.*"):
            try:
                nums.append(int(p.stem.split("_")[0]))
            except Exception:
                pass
        return max(nums) + 1 if nums else 1

    def _build_ui(self) -> None:
        top = tk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        self.info_var = tk.StringVar()
        tk.Label(top, textvariable=self.info_var, anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Label(top, text="Продукт/пачка:").pack(side=tk.LEFT, padx=(12, 4))
        self.product_var = tk.StringVar(value=self.current_product)
        self.product_entry = tk.Entry(top, textvariable=self.product_var, width=32)
        self.product_entry.pack(side=tk.LEFT)

        self.batch_var = tk.StringVar()
        tk.Label(top, textvariable=self.batch_var, fg="#0645AD", width=32, anchor="e").pack(side=tk.LEFT, padx=(10, 0))

        self.canvas = tk.Canvas(self.root, bg="#222222")
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=4)

        bottom = tk.Frame(self.root)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=8)

        tk.Button(bottom, text="Save & Next (Enter)", command=lambda: self.save_bbox(stay=False), width=18).pack(side=tk.LEFT, padx=3)
        tk.Button(bottom, text="Save Same (Ctrl+Enter)", command=lambda: self.save_bbox(stay=True), width=20).pack(side=tk.LEFT, padx=3)
        tk.Button(bottom, text="Undo last", command=self.undo_last_bbox, width=10).pack(side=tk.LEFT, padx=3)
        tk.Button(bottom, text="Skip", command=self.skip, width=8).pack(side=tk.LEFT, padx=3)
        tk.Button(bottom, text="Back", command=self.back, width=8).pack(side=tk.LEFT, padx=3)
        tk.Button(bottom, text="Reset", command=self.reset_current_bbox, width=8).pack(side=tk.LEFT, padx=3)
        tk.Button(bottom, text="Отправить пачку", command=self.upload_current_batch, width=18, bg="#dff0d8").pack(side=tk.LEFT, padx=(16, 3))
        tk.Button(bottom, text="Quit", command=self._quit, width=10).pack(side=tk.RIGHT, padx=3)

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Configure>", lambda e: self._render())

    def _bind_keys(self) -> None:
        self.root.bind("<Return>", lambda e: self.save_bbox(stay=False))
        self.root.bind("<Control-Return>", lambda e: self.save_bbox(stay=True))
        self.root.bind("<Delete>", lambda e: self.reset_current_bbox())
        self.root.bind("r", lambda e: self.reset_current_bbox())
        self.root.bind("R", lambda e: self.reset_current_bbox())
        self.root.bind("<Escape>", lambda e: self._quit())
        self.root.bind("u", lambda e: self.upload_current_batch())
        self.root.bind("U", lambda e: self.upload_current_batch())

    def _load_current(self) -> None:
        self.current_path = self.paths[self.index]
        with Image.open(self.current_path) as im:
            self.img = ImageOps.exif_transpose(im).convert("RGB")
        self.current_bbox = None
        self.start_xy = None
        self._render()
        self._update_info()

    def _update_info(self) -> None:
        if not self.current_path or not self.img:
            return
        key = str(self.current_path.resolve())
        existing = self._read_existing_bboxes_for_source(key)
        self.info_var.set(
            f"{self.index + 1}/{len(self.paths)} | {self.current_path.name} | "
            f"{self.img.width}x{self.img.height} | bbox на фото: {len(existing)}"
        )
        self.batch_var.set(f"В текущей пачке: {len(self.current_batch)} фото")

    def _render(self) -> None:
        if self.img is None:
            return
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        iw, ih = self.img.size
        self.scale = min(cw / iw, ch / ih)
        if self.scale <= 0:
            self.scale = 1.0
        new_w = max(1, int(iw * self.scale))
        new_h = max(1, int(ih * self.scale))
        resized = self.img.resize((new_w, new_h), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(resized)
        self.offset_x = (cw - new_w) // 2
        self.offset_y = (ch - new_h) // 2
        self.canvas.delete("all")
        self.saved_rect_ids.clear()
        self.current_rect_id = None
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.tk_img)

        if self.current_path:
            key = str(self.current_path.resolve())
            for bbox in self._read_existing_bboxes_for_source(key):
                self._draw_saved_bbox(bbox)
        if self.current_bbox:
            self._draw_current_bbox(self.current_bbox)

    def canvas_to_image(self, x: int, y: int) -> Tuple[int, int]:
        if self.img is None:
            return 0, 0
        ix = int((x - self.offset_x) / self.scale)
        iy = int((y - self.offset_y) / self.scale)
        ix = max(0, min(self.img.width, ix))
        iy = max(0, min(self.img.height, iy))
        return ix, iy

    def image_to_canvas(self, x: int, y: int) -> Tuple[int, int]:
        return int(x * self.scale + self.offset_x), int(y * self.scale + self.offset_y)

    def _draw_saved_bbox(self, bbox: BBox) -> None:
        x1, y1 = self.image_to_canvas(bbox.x1, bbox.y1)
        x2, y2 = self.image_to_canvas(bbox.x2, bbox.y2)
        rid = self.canvas.create_rectangle(x1, y1, x2, y2, outline="#FFD700", width=2)
        self.saved_rect_ids.append(rid)

    def _draw_current_bbox(self, bbox: BBox) -> None:
        if self.current_rect_id:
            self.canvas.delete(self.current_rect_id)
        x1, y1 = self.image_to_canvas(bbox.x1, bbox.y1)
        x2, y2 = self.image_to_canvas(bbox.x2, bbox.y2)
        self.current_rect_id = self.canvas.create_rectangle(x1, y1, x2, y2, outline="lime", width=3)

    def on_mouse_down(self, event) -> None:
        self.start_xy = self.canvas_to_image(event.x, event.y)
        self.current_bbox = None
        if self.current_rect_id:
            self.canvas.delete(self.current_rect_id)
            self.current_rect_id = None

    def on_mouse_drag(self, event) -> None:
        if not self.start_xy or self.img is None:
            return
        x1, y1 = self.start_xy
        x2, y2 = self.canvas_to_image(event.x, event.y)
        self.current_bbox = BBox(x1, y1, x2, y2).clamped(self.img.width, self.img.height)
        self._draw_current_bbox(self.current_bbox)

    def on_mouse_up(self, event) -> None:
        if not self.start_xy or self.img is None:
            return
        x1, y1 = self.start_xy
        x2, y2 = self.canvas_to_image(event.x, event.y)
        bbox = BBox(x1, y1, x2, y2).clamped(self.img.width, self.img.height)
        if bbox.valid():
            self.current_bbox = bbox
            self._draw_current_bbox(bbox)
        else:
            self.reset_current_bbox()

    def reset_current_bbox(self) -> None:
        self.current_bbox = None
        if self.current_rect_id:
            self.canvas.delete(self.current_rect_id)
            self.current_rect_id = None

    def _safe_stem(self, src: Path) -> str:
        s = "".join(ch if ch.isalnum() else "_" for ch in src.stem)
        return s[:50] or "image"

    def _get_or_create_yolo_files(self, src: Path, img: Image.Image) -> SourceMapRow:
        key = str(src.resolve())
        if key in self.source_map:
            return self.source_map[key]

        img_name = f"{self.next_yolo_index:06d}_{self._safe_stem(src)}.jpg"
        self.next_yolo_index += 1
        label_name = Path(img_name).with_suffix(".txt").name

        img.save(self.images_dir / img_name, format="JPEG", quality=95)
        (self.labels_dir / label_name).touch(exist_ok=True)

        row = SourceMapRow(key, f"images/{img_name}", f"labels/{label_name}")
        with self.sources_csv.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([row.source, row.yolo_image, row.yolo_label])
        self.source_map[key] = row
        return row

    def _read_existing_bboxes_for_source(self, source_key: str) -> List[BBox]:
        row = self.source_map.get(source_key)
        if not row or self.img is None:
            return []
        label_path = self.out / row.yolo_label
        if not label_path.exists():
            return []
        bboxes: List[BBox] = []
        w, h = self.img.size
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                xc, yc, bw, bh = map(float, parts[1:5])
            except Exception:
                continue
            x1 = int((xc - bw / 2.0) * w)
            y1 = int((yc - bh / 2.0) * h)
            x2 = int((xc + bw / 2.0) * w)
            y2 = int((yc + bh / 2.0) * h)
            b = BBox(x1, y1, x2, y2).clamped(w, h)
            if b.valid():
                bboxes.append(b)
        return bboxes

    def save_bbox(self, stay: bool) -> None:
        if self.img is None or self.current_path is None:
            return
        if self.current_bbox is None or not self.current_bbox.valid():
            messagebox.showwarning("Нет bbox", "Сначала выдели дату мышью")
            return

        bbox = self.current_bbox.clamped(self.img.width, self.img.height)
        row = self._get_or_create_yolo_files(self.current_path, self.img)
        label_path = self.out / row.yolo_label
        xc, yc, bw, bh = bbox.normalized(self.img.width, self.img.height)
        with label_path.open("a", encoding="utf-8") as f:
            f.write(f"0 {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}\n")

        self.current_batch.add(row.source)
        self._save_current_batch()
        self._save_state()
        print(f"[SAVE] {self.current_path.name} -> {row.yolo_image} bbox=({bbox.x1},{bbox.y1},{bbox.x2},{bbox.y2})")

        self.reset_current_bbox()
        if stay:
            self._render()
            self._update_info()
            return
        self.skip()

    def undo_last_bbox(self) -> None:
        if self.current_path is None:
            return
        key = str(self.current_path.resolve())
        row = self.source_map.get(key)
        if not row:
            return
        label_path = self.out / row.yolo_label
        if not label_path.exists():
            return
        lines = [ln for ln in label_path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
        if not lines:
            return
        lines = lines[:-1]
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        if not lines and key in self.current_batch:
            self.current_batch.remove(key)
            self._save_current_batch()
        self._render()
        self._update_info()
        print(f"[UNDO] removed last bbox for {self.current_path.name}")

    def skip(self) -> None:
        self.index += 1
        if self.index >= len(self.paths):
            self.index = len(self.paths) - 1
            messagebox.showinfo("Конец", "Изображения закончились")
            self._save_state()
            return
        self._save_state()
        self._load_current()

    def back(self) -> None:
        if self.index > 0:
            self.index -= 1
            self._save_state()
            self._load_current()

    def _normalize_label_text(self, label_path: Path) -> str:
        out: List[str] = []
        for raw in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = raw.strip().split()
            if len(parts) < 5:
                continue
            try:
                xc, yc, bw, bh = map(float, parts[1:5])
            except Exception:
                continue
            if bw <= 0 or bh <= 0:
                continue
            xc = min(max(xc, 0.0), 1.0)
            yc = min(max(yc, 0.0), 1.0)
            bw = min(max(bw, 0.0), 1.0)
            bh = min(max(bh, 0.0), 1.0)
            out.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        return "\n".join(out) + ("\n" if out else "")

    def _build_zip_for_sources(self, sources: List[str], zip_path: Path) -> Tuple[Path, int]:
        tmp = Path(tempfile.mkdtemp(prefix="real_photo_ds3_batch_"))
        img_out = tmp / "images"
        lbl_out = tmp / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        copied = 0
        for source_key in sources:
            row = self.source_map.get(source_key)
            if not row:
                print(f"[SKIP] source not in source_map: {source_key}")
                continue
            image_path = self.out / row.yolo_image
            label_path = self.out / row.yolo_label
            if not image_path.exists() or not label_path.exists():
                print(f"[SKIP] missing file for {source_key}")
                continue
            label_text = self._normalize_label_text(label_path)
            if not label_text.strip():
                print(f"[SKIP] empty label: {label_path}")
                continue
            copied += 1
            stem = f"frame_{copied:05d}"
            shutil.copy2(image_path, img_out / f"{stem}.jpg")
            (lbl_out / f"{stem}.txt").write_text(label_text, encoding="utf-8")

        if copied == 0:
            shutil.rmtree(tmp, ignore_errors=True)
            raise RuntimeError("В текущей пачке нет валидных размеченных изображений")

        zip_path.parent.mkdir(parents=True, exist_ok=True)
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fp in sorted(tmp.rglob("*")):
                if fp.is_file():
                    zf.write(fp, fp.relative_to(tmp).as_posix())
        shutil.rmtree(tmp, ignore_errors=True)
        return zip_path, copied

    def _login(self, session: requests.Session) -> None:
        base_url = self.args.base_url.rstrip("/")
        email = self.args.email
        password = self.args.password
        if not email or not password:
            raise RuntimeError(
                "Не задан email/password. Передай --email/--password или задай переменные "
                "PRODUCTS_DATE_BACKEND_EMAIL и PRODUCTS_DATE_BACKEND_PASSWORD."
            )
        r = session.post(f"{base_url}/api/Auth/login", json={"email": email, "password": password}, timeout=30)
        if r.status_code >= 400:
            raise RuntimeError(f"Login failed: HTTP {r.status_code}\n{r.text}")
        data = r.json()
        token = data.get("accessToken") or data.get("AccessToken")
        if not token:
            raise RuntimeError(f"В ответе login нет accessToken: {data}")
        session.headers.update({"Authorization": f"Bearer {token}"})

    def _upload_zip(self, session: requests.Session, zip_path: Path, comment: str) -> Dict:
        base_url = self.args.base_url.rstrip("/")
        with zip_path.open("rb") as f:
            r = session.post(
                f"{base_url}/api/error-reports/upload-dataset",
                data={"comment": comment},
                files={"datasetZip": (zip_path.name, f, "application/zip")},
                timeout=self.args.upload_timeout,
            )
        if r.status_code >= 400:
            raise RuntimeError(f"Upload failed: HTTP {r.status_code}\n{r.text}")
        return r.json()

    def _approve_report(self, session: requests.Session, report_id: int) -> None:
        base_url = self.args.base_url.rstrip("/")
        r = session.put(f"{base_url}/api/admin/error-reports/{report_id}/approve", json={"approved": True}, timeout=30)
        if r.status_code >= 400:
            raise RuntimeError(f"Approve failed: HTTP {r.status_code}\n{r.text}")

    def _next_batch_id(self) -> int:
        max_id = 0
        if self.batch_log_csv.exists():
            with self.batch_log_csv.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    try:
                        max_id = max(max_id, int(r.get("batch_id") or 0))
                    except Exception:
                        pass
        return max_id + 1

    def upload_current_batch(self) -> None:
        if not self.current_batch:
            messagebox.showinfo("Пачка пустая", "В текущей пачке пока нет размеченных изображений")
            return
        product = self.product_var.get().strip()
        if not product:
            product = simpledialog.askstring("Название продукта", "Как назвать эту пачку/продукт?", parent=self.root) or "без названия"
            self.product_var.set(product)
        count = len(self.current_batch)
        if not messagebox.askyesno("Отправить пачку", f"Отправить текущую пачку '{product}'?\nИзображений: {count}"):
            return

        batch_id = self._next_batch_id()
        zip_path = Path(tempfile.gettempdir()) / f"real_photo_ds3_product_batch_{batch_id:03d}.zip"
        try:
            zip_path, copied = self._build_zip_for_sources(sorted(self.current_batch), zip_path)
            size_mb = zip_path.stat().st_size / (1024 * 1024)
            comment = f"{self.args.comment}; продукт: {product}; batch {batch_id}; images={copied}"
            print(f"[ZIP] {zip_path} | {copied} images | {size_mb:.2f} MB")

            session = requests.Session()
            self._login(session)
            print("[AUTH] OK")
            result = self._upload_zip(session, zip_path, comment)
            print("[UPLOAD] OK")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            report_id = int(result.get("reportId") or result.get("ReportId"))
            if self.args.approve:
                self._approve_report(session, report_id)
                print(f"[APPROVE] reportId={report_id}")

            with self.batch_log_csv.open("a", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow([time.strftime("%Y-%m-%d %H:%M:%S"), batch_id, report_id, product, copied, comment])

            self.current_batch.clear()
            self._save_current_batch()
            self._save_state()
            self._update_info()
            messagebox.showinfo("Готово", f"Пачка отправлена. reportId={report_id}")
        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")
            messagebox.showerror("Ошибка отправки", str(e))

    def _quit(self) -> None:
        self._save_state()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Label real photos and upload product-sized batches to ProductsDateAPI")
    p.add_argument("--src", default=r"D:\diplom\datasets\real_photo_ds3", help="Папка с исходными неразмеченными фото")
    p.add_argument("--out", default=r"D:\diplom\datasets\real_photo_ds3_labeled", help="Куда сохранять размеченный YOLO dataset")
    p.add_argument("--base-url", default=os.getenv("PRODUCTS_DATE_BACKEND_URL", "http://111.88.146.2:5099"))
    p.add_argument("--email", default=os.getenv("PRODUCTS_DATE_BACKEND_EMAIL", ""))
    p.add_argument("--password", default=os.getenv("PRODUCTS_DATE_BACKEND_PASSWORD", ""))
    p.add_argument("--comment", default="Реальные фотографии, размеченные вручную по пачкам продуктов")
    p.add_argument("--approve", action="store_true", help="После upload сразу отметить отчёт approved")
    p.add_argument("--upload-timeout", type=int, default=900)
    p.add_argument("--start-index", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    app = ProductBatchLabeler(args)
    app.run()


if __name__ == "__main__":
    main()
