import cv2
import numpy as np
import time

VIDEO_SOURCE = "medias/IMG_5068.MOV"   # или 0 для вебки
MAX_WIDTH = 640
DEBUG_EVERY_N = 10

# фильтры под "мелкий текст, почти ч/б"
MIN_AREA = 300
MAX_AREA = 12000

MIN_H = 10
MAX_H = 80

MIN_ASPECT = 0.4
MAX_ASPECT = 8.0

MAX_MEAN_SAT = 110   # порог насыщенности (0..255)

MAX_BOXES_SHOWN = 5  # максимум зелёных квадратов на экране


def find_date_like_regions(gray, color_bgr):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    bin_img = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15, 10
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    morph = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(
        morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    H, W = gray.shape[:2]
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)

    candidates = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < MIN_AREA or area > MAX_AREA:
            continue

        if w < 5 or h < 5:
            continue

        aspect = w / float(h)
        if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
            continue

        if h < MIN_H or h > MAX_H:
            continue

        # чуть режем самые верх-низ
        if y < H * 0.05 or y > H * 0.95:
            continue

        roi_bin = bin_img[y:y + h, x:x + w]
        white = cv2.countNonZero(roi_bin)
        ink_ratio = white / float(area)
        if ink_ratio < 0.08 or ink_ratio > 0.85:
            continue

        roi_hsv = hsv[y:y + h, x:x + w]
        mean_sat = float(np.mean(roi_hsv[:, :, 1]))
        if mean_sat > MAX_MEAN_SAT:
            continue

        candidates.append((x, y, w, h))

    return candidates, bin_img


def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"❌ Не удалось открыть видеоисточник: {VIDEO_SOURCE}")
        return

    print("✅ Видеоисточник открыт. ESC — выход.")
    frame_idx = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("🏁 Видео закончилось или камера недоступна.")
            break

        h, w = frame.shape[:2]
        if w > MAX_WIDTH:
            scale = MAX_WIDTH / float(w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            h, w = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        t0 = time.time()
        candidates, bin_img = find_date_like_regions(gray, frame)
        dt = (time.time() - t0) * 1000.0

        # --- выбираем максимум MAX_BOXES_SHOWN кандидатов, ближайших к центру ---
        cx, cy = w / 2.0, h / 2.0
        scored = []
        for (x, y, cw, ch) in candidates:
            bx = x + cw / 2.0
            by = y + ch / 2.0
            dist2 = (bx - cx) ** 2 + (by - cy) ** 2
            scored.append((dist2, (x, y, cw, ch)))

        scored.sort(key=lambda s: s[0])
        top = [bb for _, bb in scored[:MAX_BOXES_SHOWN]]

        # рисуем только зелёные квадраты для выбранных кандидатов
        for (x, y, cw, ch) in top:
            cv2.rectangle(frame, (x, y), (x + cw, y + ch), (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"candidates: {len(candidates)}  shown: {len(top)}  time: {dt:.1f}ms",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

        if frame_idx % DEBUG_EVERY_N == 0:
            print(f"Кадр {frame_idx}: всего {len(candidates)}, показываем {len(top)}, {dt:.1f} мс")

        cv2.imshow("Frame (подозрительные зоны)", frame)
        # cv2.imshow("Binary", bin_img)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        frame_idx += 1

    total_time = time.time() - t_start
    print(f"\n⏱ Кадров: {frame_idx}, время: {total_time:.1f} сек")
    if frame_idx > 0:
        print(f"Средний FPS: {frame_idx / total_time:.1f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
