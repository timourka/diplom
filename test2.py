import cv2
import numpy as np
import time

# --- НАСТРОЙКИ ---

# Если хочешь читать из файла:
VIDEO_SOURCE = "medias/IMG_5066.MOV"   # или "video.mp4"

# Если хочешь с камеры, раскомментируй это:
# VIDEO_SOURCE = 0  # 0 — первая вебка

MAX_WIDTH = 640       # до какой ширины сжимать кадр
DEBUG_EVERY_N = 10    # каждые N кадров выводить отладку


def find_date_like_regions(gray):
    """
    На входе — серый кадр (уменьшенный).
    На выходе — список прямоугольников (x, y, w, h),
    которые выглядят как мелкие горизонтальные текстовые зоны.
    """
    # Лёгкое размытие, чтобы убрать шум
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Бинаризация (текст -> белый, фон -> чёрный, инверсия)
    # Параметры можно крутить
    bin_img = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15, 10
    )

    # Морфология: склеиваем мелкие символы в полоски
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    morph = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Находим контуры предполагаемых "текстовых блоков"
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = gray.shape[:2]
    candidates = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 150 or area > 6000:
            continue

        aspect = w / float(h)
        if aspect < 2.0 or aspect > 20.0:
            # Слишком квадратное или слишком длинное
            continue

        # Фильтр по высоте (мелкий текст)
        if h < 8 or h > 40:
            continue

        # Можно искать только в "полезной" вертикальной зоне (не самое небо и не самый низ)
        if y < H * 0.1 or y > H * 0.9:
            continue

        # Проверка "насыщенности": сколько белых пикселей внутри (символы) относительно площади
        roi = bin_img[y:y + h, x:x + w]
        white = cv2.countNonZero(roi)
        ink_ratio = white / float(area)

        # Слишком пусто или слишком залито — отбрасываем
        if ink_ratio < 0.15 or ink_ratio > 0.85:
            continue

        candidates.append((x, y, w, h))

    return candidates, bin_img


def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"❌ Не удалось открыть источник видео: {VIDEO_SOURCE}")
        return

    print("✅ Видеоисточник открыт. Нажми ESC, чтобы выйти.")
    frame_idx = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("🏁 Видео закончилось или камера недоступна.")
            break

        # Уменьшаем кадр по ширине
        h, w = frame.shape[:2]
        scale = 1.0
        if w > MAX_WIDTH:
            scale = MAX_WIDTH / float(w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            h, w = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        t0 = time.time()
        candidates, bin_img = find_date_like_regions(gray)
        dt = (time.time() - t0) * 1000  # мс

        # Рисуем рамки: зелёные — обычные кандидаты,
        # красный — "лучший" (просто с максимальным отношением сторон).
        best_idx = None
        best_aspect = 0
        for i, (x, y, cw, ch) in enumerate(candidates):
            aspect = cw / float(ch)
            if aspect > best_aspect:
                best_aspect = aspect
                best_idx = i

        for i, (x, y, cw, ch) in enumerate(candidates):
            color = (0, 255, 0)
            if i == best_idx:
                color = (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + cw, y + ch), color, 2)

        # Небольшой текст поверх кадра
        cv2.putText(
            frame,
            f"candidates: {len(candidates)}  time: {dt:.1f}ms",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

        # Периодически печатаем в консоль
        if frame_idx % DEBUG_EVERY_N == 0:
            print(f"Кадр {frame_idx}: кандидатов {len(candidates)}, обработка {dt:.1f} мс")

        cv2.imshow("Frame (подозрительные зоны подсвечены)", frame)
        # Можно также посмотреть бинарную картинку:
        # cv2.imshow("Binary", bin_img)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

        frame_idx += 1

    total_time = time.time() - t_start
    print(f"\n⏱ Всего кадров: {frame_idx}, время: {total_time:.1f} сек")
    if frame_idx > 0:
        print(f"Средний FPS (с учётом обработки): {frame_idx / total_time:.1f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
