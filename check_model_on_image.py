from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Проверка YOLO/TFLite модели на одном кадре с визуализацией.')
    p.add_argument('--model', required=True, help='Путь к модели: .pt, .onnx, .tflite и т.п.')
    p.add_argument('--image', required=True, help='Путь к изображению.')
    p.add_argument('--label', default=None, help='Путь к YOLO label .txt для наложения GT.')
    p.add_argument('--imgsz', type=int, default=640, help='Размер инференса.')
    p.add_argument('--conf', type=float, default=0.05, help='Порог уверенности.')
    p.add_argument('--iou', type=float, default=0.45, help='NMS IoU.')
    p.add_argument('--device', default='cpu', help='Устройство для PT/ONNX. Для TFLite обычно cpu.')
    p.add_argument('--save', default='prediction_debug.png', help='Куда сохранить результат.')
    p.add_argument('--class-name', default='object', help='Имя класса для подписи GT.')
    return p.parse_args()


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f'Не удалось прочитать изображение: {path}')
    return img


def draw_gt(img: np.ndarray, label_path: Path, class_name: str) -> None:
    if not label_path.exists():
        print(f'[WARN] label не найден: {label_path}')
        return
    h, w = img.shape[:2]
    for line in label_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            print(f'[WARN] пропускаю кривую строку label: {line}')
            continue
        _, xc, yc, bw, bh = map(float, parts)
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'GT: {class_name}', (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def draw_preds(img: np.ndarray, boxes, names: dict[int, str]) -> None:
    if boxes is None or len(boxes) == 0:
        print('[INFO] Предсказаний нет.')
        return
    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0].cpu().numpy())
        cls = int(box.cls[0].cpu().numpy())
        x1, y1, x2, y2 = xyxy.tolist()
        label = f'PRED: {names.get(cls, str(cls))} {conf:.3f}'
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, label, (x1, max(20, y2 + 20 if y1 < 25 else y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    image_path = Path(args.image)
    label_path = Path(args.label) if args.label else None
    save_path = Path(args.save)

    print(f'[INFO] model = {model_path}')
    print(f'[INFO] image = {image_path}')
    if label_path:
        print(f'[INFO] label = {label_path}')

    model = YOLO(str(model_path))
    res = model.predict(
        source=str(image_path),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=True,
        save=False,
    )[0]

    print(f'[INFO] names = {res.names}')
    if res.boxes is None or len(res.boxes) == 0:
        print('[INFO] Модель ничего не нашла.')
    else:
        print(f'[INFO] boxes count = {len(res.boxes)}')
        for i, b in enumerate(res.boxes, start=1):
            xyxy = b.xyxy[0].cpu().numpy().round(2).tolist()
            conf = float(b.conf[0].cpu().numpy())
            cls = int(b.cls[0].cpu().numpy())
            print(f'  #{i}: cls={cls} conf={conf:.4f} xyxy={xyxy}')

    img = load_image(image_path)
    if label_path:
        draw_gt(img, label_path, args.class_name)
    draw_preds(img, res.boxes, res.names)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(save_path), img)
    if not ok:
        raise RuntimeError(f'Не удалось сохранить {save_path}')
    print(f'[INFO] Сохранено: {save_path.resolve()}')


if __name__ == '__main__':
    main()
