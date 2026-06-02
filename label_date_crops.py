#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Простая ручная разметка crop-ов даты.

Что делает:
- берет изображения из папки crops_images;
- показывает по одному изображению;
- ты вводишь строку даты;
- по кнопке Save копирует изображение в date_reader_real/images;
- добавляет строку в date_reader_real/labels.csv.

Формат labels.csv:
image,text,source
images/000001.jpg,02.12.2026,D:/diplom/crops_images/xxx.jpg
"""

import argparse
import csv
import os
import shutil
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

try:
    from PIL import Image, ImageTk
except ImportError:
    raise SystemExit("Не установлен Pillow. Установи: pip install pillow")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(src: Path):
    return sorted([p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS])


def read_done_sources(labels_csv: Path):
    done = set()
    if not labels_csv.exists():
        return done
    with labels_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = row.get("source") or ""
            if src:
                done.add(str(Path(src)))
    return done


def next_output_name(images_dir: Path, ext: str):
    existing = []
    for p in images_dir.iterdir() if images_dir.exists() else []:
        if p.is_file() and p.stem.isdigit():
            try:
                existing.append(int(p.stem))
            except ValueError:
                pass
    idx = max(existing, default=0) + 1
    return f"{idx:06d}{ext.lower()}"


class LabelApp:
    def __init__(self, root, src: Path, out: Path, resume: bool = True):
        self.root = root
        self.src = src
        self.out = out
        self.images_dir = out / "images"
        self.labels_csv = out / "labels.csv"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.out.mkdir(parents=True, exist_ok=True)

        all_images = collect_images(src)
        if resume:
            done = read_done_sources(self.labels_csv)
            self.images = [p for p in all_images if str(p) not in done]
        else:
            self.images = all_images

        self.index = 0
        self.current_photo = None

        self.root.title("Date crop labeler")
        self.root.geometry("1000x700")

        self.info = tk.Label(root, text="", font=("Arial", 12))
        self.info.pack(pady=5)

        self.image_label = tk.Label(root, bg="#222")
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        bottom = tk.Frame(root)
        bottom.pack(fill=tk.X, padx=10, pady=8)

        tk.Label(bottom, text="Дата:", font=("Arial", 12)).pack(side=tk.LEFT)
        self.entry = tk.Entry(bottom, font=("Arial", 16))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.entry.bind("<Return>", lambda e: self.save())

        tk.Button(bottom, text="Save", command=self.save, width=12, height=2).pack(side=tk.LEFT, padx=4)
        tk.Button(bottom, text="Skip", command=self.skip, width=12, height=2).pack(side=tk.LEFT, padx=4)
        tk.Button(bottom, text="Back", command=self.back, width=12, height=2).pack(side=tk.LEFT, padx=4)
        tk.Button(bottom, text="Exit", command=root.destroy, width=12, height=2).pack(side=tk.LEFT, padx=4)

        hint = tk.Label(
            root,
            text="Enter = сохранить | Skip = пропустить | Back = назад. Пустую дату сохранить нельзя.",
            font=("Arial", 10),
        )
        hint.pack(pady=4)

        self.ensure_csv_header()
        self.show_current()

    def ensure_csv_header(self):
        if not self.labels_csv.exists():
            with self.labels_csv.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["image", "text", "source"])

    def show_current(self):
        if not self.images:
            self.info.config(text="Нет изображений для разметки")
            self.image_label.config(image="")
            return
        if self.index >= len(self.images):
            messagebox.showinfo("Готово", "Все изображения обработаны")
            self.root.destroy()
            return

        path = self.images[self.index]
        self.info.config(text=f"{self.index + 1}/{len(self.images)}: {path.name}")
        self.entry.delete(0, tk.END)
        self.entry.focus_set()

        img = Image.open(path).convert("RGB")
        max_w, max_h = 940, 500
        img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        self.current_photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.current_photo)

    def save(self):
        if not self.images or self.index >= len(self.images):
            return
        text = self.entry.get().strip()
        if not text:
            messagebox.showwarning("Пустая дата", "Введи дату или нажми Skip")
            return

        src_path = self.images[self.index]
        out_name = next_output_name(self.images_dir, src_path.suffix)
        out_path = self.images_dir / out_name
        shutil.copy2(src_path, out_path)

        rel_img = f"images/{out_name}"
        with self.labels_csv.open("a", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([rel_img, text, str(src_path)])

        self.index += 1
        self.show_current()

    def skip(self):
        self.index += 1
        self.show_current()

    def back(self):
        if self.index > 0:
            self.index -= 1
            self.show_current()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="crops_images", help="Папка с crop-изображениями")
    parser.add_argument("--out", default="date_reader_real", help="Папка результата")
    parser.add_argument("--no-resume", action="store_true", help="Не пропускать уже сохраненные source")
    args = parser.parse_args()

    src = Path(args.src).resolve()
    out = Path(args.out).resolve()

    if not src.exists():
        raise SystemExit(f"Папка не найдена: {src}")

    root = tk.Tk()
    LabelApp(root, src=src, out=out, resume=not args.no_resume)
    root.mainloop()


if __name__ == "__main__":
    main()
