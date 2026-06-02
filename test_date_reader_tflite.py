import csv
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


MODEL_PATH = r"D:\diplom\date_reader_work_v5_gpu\date_reader.tflite"
META_PATH = r"D:\diplom\date_reader_work_v5_gpu\date_reader_meta.json"
TEST_DIR = Path(r"D:\diplom\date_reader_test")
IMAGES_DIR = TEST_DIR / "images"
LABELS_CSV = TEST_DIR / "labels.csv"


def levenshtein(a: str, b: str) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1]


def cer(pred: str, gt: str) -> float:
    if len(gt) == 0:
        return 0.0 if pred == "" else 1.0
    return levenshtein(pred, gt) / len(gt)


def preprocess(image_path: Path) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(image_path)

    img = cv2.resize(img, (256, 48), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = img[None, :, :, None]  # 1 x 48 x 256 x 1
    return img


def ctc_greedy_decode(logits: np.ndarray, alphabet: list[str], blank_index: int) -> str:
    # logits: 64 x num_classes
    ids = np.argmax(logits, axis=-1)

    result = []
    prev = None
    for idx in ids:
        idx = int(idx)
        if idx != blank_index and idx != prev:
            if 0 <= idx < len(alphabet):
                result.append(alphabet[idx])
        prev = idx

    return "".join(result)


def main():
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Подстрой под свой meta, если ключи называются иначе.
    alphabet = meta.get("alphabet") or meta.get("chars")
    if alphabet is None:
        raise KeyError("В meta.json не найден alphabet/chars")

    blank_index = meta.get("blank_index", len(alphabet))

    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    total = 0
    exact_ok = 0
    cer_sum = 0.0
    errors = []

    with open(LABELS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            filename = row["filename"]
            gt = row["text"].strip()

            x = preprocess(IMAGES_DIR / filename)

            interpreter.set_tensor(input_details[0]["index"], x)
            interpreter.invoke()

            y = interpreter.get_tensor(output_details[0]["index"])[0]  # 64 x 14
            pred = ctc_greedy_decode(y, alphabet, blank_index).strip()

            total += 1
            is_exact = pred == gt
            exact_ok += int(is_exact)
            sample_cer = cer(pred, gt)
            cer_sum += sample_cer

            if not is_exact:
                errors.append((filename, gt, pred, sample_cer))

    exact = exact_ok / total if total else 0
    mean_cer = cer_sum / total if total else 0

    print(f"TOTAL: {total}")
    print(f"EXACT: {exact:.3f}")
    print(f"CER:   {mean_cer:.3f}")
    print()
    print("Ошибки:")
    for filename, gt, pred, sample_cer in errors[:50]:
        print(f"{filename}: GT={gt} | PRED={pred} | CER={sample_cer:.3f}")


if __name__ == "__main__":
    main()