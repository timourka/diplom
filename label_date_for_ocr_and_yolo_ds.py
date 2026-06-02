#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Разметка фото с датами сразу для двух задач:
1) OCR/date_reader: сохраняет кроп даты + labels.csv с текстом даты.
2) YOLO/date detector: копирует ОРИГИНАЛЬНОЕ изображение в datasets/real_photo_ds2/images
   и сохраняет bbox в datasets/real_photo_ds2/labels/*.txt.

Запуск:
  python label_date_for_ocr_and_yolo_ds.py --src D:\\diplom\\crops_images --ocr-out D:\\diplom\\date_reader_real --yolo-out D:\\diplom\\datasets\\real_photo_ds2

Управление:
  ЛКМ + протянуть      выделить дату
  Enter / Save & Next  сохранить bbox+текст и перейти к следующему изображению
  Ctrl+Enter / Save Same сохранить bbox+текст и остаться на том же изображении
  Skip                 пропустить изображение
  Back                 вернуться назад
  R или Delete         сбросить выделение
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def normalized(self, w: int, h: int) -> Tuple[float, float, float, float]:
        x1, x2 = sorted((max(0, self.x1), min(w, self.x2)))
        y1, y2 = sorted((max(0, self.y1), min(h, self.y2)))
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        xc = x1 + bw / 2.0
        yc = y1 + bh / 2.0
        return xc / w, yc / h, bw / w, bh / h

    def clamped(self, w: int, h: int) -> "BBox":
        x1, x2 = sorted((self.x1, self.x2))
        y1, y2 = sorted((self.y1, self.y2))
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        return BBox(x1, y1, x2, y2)

    def valid(self, min_size: int = 3) -> bool:
        return abs(self.x2 - self.x1) >= min_size and abs(self.y2 - self.y1) >= min_size


class DateLabelApp:
    def __init__(self, src: Path, ocr_out: Path, yolo_out: Path, start_index: int = 0):
        self.src = src
        self.ocr_out = ocr_out
        self.yolo_out = yolo_out

        self.ocr_images_dir = self.ocr_out / "images"
        self.yolo_images_dir = self.yolo_out / "images"
        self.yolo_labels_dir = self.yolo_out / "labels"
        self.ocr_images_dir.mkdir(parents=True, exist_ok=True)
        self.yolo_images_dir.mkdir(parents=True, exist_ok=True)
        self.yolo_labels_dir.mkdir(parents=True, exist_ok=True)

        self.ocr_csv = self.ocr_out / "labels.csv"
        self.source_map_csv = self.yolo_out / "sources.csv"
        self._ensure_csvs()
        self.source_to_yolo_name = self._load_source_map()

        self.paths = sorted([p for p in self.src.rglob("*") if p.suffix.lower() in IMAGE_EXTS])
        if not self.paths:
            raise SystemExit(f"Нет изображений в {self.src}")

        self.index = max(0, min(start_index, len(self.paths) - 1))
        self.ocr_counter = self._next_ocr_index()
        self.yolo_counter = self._next_yolo_index()

        self.root = tk.Tk()
        self.root.title("Date crop labeler: OCR + YOLO dataset")
        self.root.geometry("1200x850")

        self.img: Optional[Image.Image] = None
        self.tk_img = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.rect_id = None
        self.start_xy = None
        self.current_bbox: Optional[BBox] = None

        self._build_ui()
        self._bind_keys()
        self._load_current()

    def _ensure_csvs(self) -> None:
        if not self.ocr_csv.exists():
            with self.ocr_csv.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["image", "text", "source", "x1", "y1", "x2", "y2"])
        if not self.source_map_csv.exists():
            with self.source_map_csv.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["source", "yolo_image", "yolo_label"])
        self._write_data_yaml()

    def _write_data_yaml(self) -> None:
        yaml_text = (
            f"path: {self.yolo_out.as_posix()}\n"
            f"train: images\n"
            f"val: images\n"
            f"names:\n"
            f"  0: expiry_date\n"
        )
        (self.yolo_out / "data.yaml").write_text(yaml_text, encoding="utf-8")

    def _load_source_map(self) -> Dict[str, str]:
        result: Dict[str, str] = {}
        if not self.source_map_csv.exists():
            return result
        with self.source_map_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                src = row.get("source", "")
                img = row.get("yolo_image", "")
                if src and img:
                    result[src] = img
        return result

    def _next_ocr_index(self) -> int:
        nums = []
        for p in self.ocr_images_dir.glob("*.*"):
            try:
                nums.append(int(p.stem.split("_")[0]))
            except ValueError:
                pass
        return (max(nums) + 1) if nums else 1

    def _next_yolo_index(self) -> int:
        nums = []
        for p in self.yolo_images_dir.glob("*.*"):
            try:
                nums.append(int(p.stem.split("_")[0]))
            except ValueError:
                pass
        return (max(nums) + 1) if nums else 1

    def _build_ui(self) -> None:
        top = tk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        self.info_var = tk.StringVar()
        tk.Label(top, textvariable=self.info_var, anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Label(top, text="Дата:").pack(side=tk.LEFT, padx=(12, 4))
        self.text_entry = tk.Entry(top, width=25, font=("Arial", 14))
        self.text_entry.pack(side=tk.LEFT)

        self.canvas = tk.Canvas(self.root, bg="#222")
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=4)

        bottom = tk.Frame(self.root)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=8)

        tk.Button(bottom, text="Save & Next (Enter)", command=lambda: self.save(stay=False), width=18).pack(side=tk.LEFT, padx=4)
        tk.Button(bottom, text="Save Same (Ctrl+Enter)", command=lambda: self.save(stay=True), width=20).pack(side=tk.LEFT, padx=4)
        tk.Button(bottom, text="Skip", command=self.skip, width=10).pack(side=tk.LEFT, padx=4)
        tk.Button(bottom, text="Back", command=self.back, width=10).pack(side=tk.LEFT, padx=4)
        tk.Button(bottom, text="Reset (R/Delete)", command=self.reset_bbox, width=14).pack(side=tk.LEFT, padx=4)
        tk.Button(bottom, text="Quit", command=self.root.destroy, width=10).pack(side=tk.RIGHT, padx=4)

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Configure>", lambda e: self._render())

    def _bind_keys(self) -> None:
        self.root.bind("<Return>", lambda e: self.save(stay=False))
        self.root.bind("<Control-Return>", lambda e: self.save(stay=True))
        self.root.bind("<Delete>", lambda e: self.reset_bbox())
        self.root.bind("r", lambda e: self.reset_bbox())
        self.root.bind("R", lambda e: self.reset_bbox())
        self.root.bind("<Escape>", lambda e: self.root.destroy())

    def _load_current(self) -> None:
        path = self.paths[self.index]
        self.img = Image.open(path).convert("RGB")
        self.current_bbox = None
        self.start_xy = None
        self.text_entry.delete(0, tk.END)
        self._render()
        self.info_var.set(f"{self.index + 1}/{len(self.paths)} | {path} | size={self.img.width}x{self.img.height}")
        self.text_entry.focus_set()

    def _render(self) -> None:
        if self.img is None:
            return
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        iw, ih = self.img.size
        self.scale = min(cw / iw, ch / ih, 1.0 if iw < cw and ih < ch else min(cw / iw, ch / ih))
        new_w = max(1, int(iw * self.scale))
        new_h = max(1, int(ih * self.scale))
        resized = self.img.resize((new_w, new_h), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(resized)
        self.offset_x = (cw - new_w) // 2
        self.offset_y = (ch - new_h) // 2
        self.canvas.delete("all")
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.tk_img)
        if self.current_bbox:
            self._draw_bbox(self.current_bbox)

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

    def _draw_bbox(self, bbox: BBox) -> None:
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        x1, y1 = self.image_to_canvas(bbox.x1, bbox.y1)
        x2, y2 = self.image_to_canvas(bbox.x2, bbox.y2)
        self.rect_id = self.canvas.create_rectangle(x1, y1, x2, y2, outline="lime", width=3)

    def on_mouse_down(self, event) -> None:
        self.start_xy = self.canvas_to_image(event.x, event.y)
        self.current_bbox = None
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None

    def on_mouse_drag(self, event) -> None:
        if not self.start_xy or self.img is None:
            return
        x1, y1 = self.start_xy
        x2, y2 = self.canvas_to_image(event.x, event.y)
        self.current_bbox = BBox(x1, y1, x2, y2).clamped(self.img.width, self.img.height)
        self._draw_bbox(self.current_bbox)

    def on_mouse_up(self, event) -> None:
        if not self.start_xy or self.img is None:
            return
        x1, y1 = self.start_xy
        x2, y2 = self.canvas_to_image(event.x, event.y)
        bbox = BBox(x1, y1, x2, y2).clamped(self.img.width, self.img.height)
        if bbox.valid():
            self.current_bbox = bbox
            self._draw_bbox(bbox)
        else:
            self.current_bbox = None
            if self.rect_id:
                self.canvas.delete(self.rect_id)
                self.rect_id = None

    def reset_bbox(self) -> None:
        self.current_bbox = None
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None

    def _get_or_create_yolo_image_name(self, src_path: Path) -> str:
        key = str(src_path.resolve())
        if key in self.source_to_yolo_name:
            return self.source_to_yolo_name[key]

        ext = src_path.suffix.lower()
        safe_stem = "".join(ch if ch.isalnum() else "_" for ch in src_path.stem)[:40]
        img_name = f"{self.yolo_counter:06d}_{safe_stem}{ext}"
        self.yolo_counter += 1

        shutil.copy2(src_path, self.yolo_images_dir / img_name)
        label_name = Path(img_name).with_suffix(".txt").name
        (self.yolo_labels_dir / label_name).touch(exist_ok=True)

        with self.source_map_csv.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([key, f"images/{img_name}", f"labels/{label_name}"])
        self.source_to_yolo_name[key] = f"images/{img_name}"
        return f"images/{img_name}"

    def save(self, stay: bool) -> None:
        if self.img is None or self.current_bbox is None:
            messagebox.showwarning("Нет bbox", "Сначала выдели область даты мышью")
            return
        text = self.text_entry.get().strip()
        if not text:
            messagebox.showwarning("Нет текста", "Введи текст даты")
            return

        src_path = self.paths[self.index]
        bbox = self.current_bbox.clamped(self.img.width, self.img.height)
        if not bbox.valid():
            messagebox.showwarning("Плохой bbox", "Выделение слишком маленькое")
            return

        # 1) OCR crop + labels.csv
        crop = self.img.crop((bbox.x1, bbox.y1, bbox.x2, bbox.y2))
        ocr_name = f"{self.ocr_counter:06d}.jpg"
        self.ocr_counter += 1
        crop.save(self.ocr_images_dir / ocr_name, quality=95)
        with self.ocr_csv.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([f"images/{ocr_name}", text, str(src_path), bbox.x1, bbox.y1, bbox.x2, bbox.y2])

        # 2) YOLO full original image + label
        yolo_img_rel = self._get_or_create_yolo_image_name(src_path)
        yolo_img_name = Path(yolo_img_rel).name
        label_path = self.yolo_labels_dir / Path(yolo_img_name).with_suffix(".txt").name
        xc, yc, bw, bh = bbox.normalized(self.img.width, self.img.height)
        with label_path.open("a", encoding="utf-8") as f:
            f.write(f"0 {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}\n")

        print(f"[SAVE] OCR images/{ocr_name} text='{text}' | YOLO {yolo_img_rel} bbox={bbox}")

        self.reset_bbox()
        if not stay:
            self.index += 1
            if self.index >= len(self.paths):
                messagebox.showinfo("Готово", "Изображения закончились")
                self.root.destroy()
                return
            self._load_current()
        else:
            self.text_entry.delete(0, tk.END)
            self.text_entry.focus_set()

    def skip(self) -> None:
        self.index += 1
        if self.index >= len(self.paths):
            messagebox.showinfo("Готово", "Изображения закончились")
            self.root.destroy()
            return
        self._load_current()

    def back(self) -> None:
        if self.index > 0:
            self.index -= 1
            self._load_current()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Папка с исходными полными изображениями")
    parser.add_argument("--ocr-out", default="date_reader_real", help="Куда сохранять OCR crop-ы и labels.csv")
    parser.add_argument("--yolo-out", default="datasets/real_photo_ds2", help="Куда сохранять YOLO images/labels/data.yaml")
    parser.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args()

    app = DateLabelApp(Path(args.src), Path(args.ocr_out), Path(args.yolo_out), args.start_index)
    app.run()


if __name__ == "__main__":
    main()
