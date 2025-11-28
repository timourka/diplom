import cv2
import easyocr
import re
import time

# --- НАСТРОЙКИ ---
VIDEO_PATH = "medias/IMG_5068.MOV"      # или .mp4
FRAME_STEP = 5                # обрабатывать каждый 5-й кадр
MIN_CONF = 0.4                # минимальная уверенность OCR
DEBUG_SHOW_TEXT = True        # печатать найденные OCR-тексты
DEBUG_EVERY_N_FRAMES = 10     # как часто печатать прогресс (по обработанным кадрам)

# --- OCR и шаблон дат ---
reader = easyocr.Reader(['ru', 'en'])

date_pattern = re.compile(
    r'(0[1-9]|[12][0-9]|3[01])[.\-/](0[1-9]|1[0-2])[.\-/](20\d{2}|\d{2})'
)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"❌ Не удалось открыть видео: {VIDEO_PATH}")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS) or 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
duration_sec = total_frames / fps if fps > 0 else 0

print("✅ Видео открыто")
print(f"  Кадров: {total_frames}")
print(f"  FPS: {fps:.2f}")
print(f"  Длительность: ~{duration_sec:.1f} сек")
print()

frame_idx = 0
processed_frames = 0
found = False
start_time_all = time.time()

try:
    while cap.isOpened() and not found:
        ret, frame = cap.read()
        if not ret:
            print("🏁 Видео закончилось, дату не нашли.")
            break

        # пропускаем лишние кадры
        if frame_idx % FRAME_STEP != 0:
            frame_idx += 1
            continue

        processed_frames += 1

        # грубая оценка прогресса
        progress = (frame_idx / total_frames * 100) if total_frames > 0 else 0
        current_time_sec = frame_idx / fps if fps > 0 else 0

        if processed_frames % DEBUG_EVERY_N_FRAMES == 0:
            print(f"[{processed_frames} обраб. кадров] "
                  f"кадр {frame_idx}/{total_frames} "
                  f"({progress:.1f}%, t≈{current_time_sec:.1f}с)")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        t0 = time.time()
        results = reader.readtext(gray, detail=1)  # [(bbox, text, conf), ...]
        t1 = time.time()

        if processed_frames % DEBUG_EVERY_N_FRAMES == 0:
            print(f"  ⏱ OCR занял: {(t1 - t0):.2f} сек, найдено фрагментов: {len(results)}")

        if DEBUG_SHOW_TEXT and processed_frames % DEBUG_EVERY_N_FRAMES == 0:
            # покажем пару первых распознанных строк
            sample_texts = [r[1] for r in results[:3]]
            print("  Примеры текста:", sample_texts)

        # поиск даты в распознанном тексте
        for bbox, text, conf in results:
            match = date_pattern.search(text)
            if match and conf > MIN_CONF:
                date_text = match.group()
                total_time = time.time() - start_time_all
                print("\n🎉 Найдена дата!")
                print(f"  Текст: {text!r}")
                print(f"  Дата: {date_text}")
                print(f"  Кадр: {frame_idx}/{total_frames} "
                      f"(t≈{current_time_sec:.1f}с, прогресс {progress:.1f}%)")
                print(f"  Уверенность OCR: {conf:.2f}")
                print(f"  Общее время работы: {total_time:.1f} сек")

                # при желании показать кадр
                cv2.imshow("Found expiration date", frame)
                cv2.waitKey(0)
                found = True
                break

        frame_idx += 1

except KeyboardInterrupt:
    print("\n⏹ Остановлено пользователем (Ctrl+C).")

finally:
    cap.release()
    cv2.destroyAllWindows()
    if not found:
        total_time = time.time() - start_time_all
        print(f"\n⏱ Скрипт закончил работу. Дату не нашли. Время: {total_time:.1f} сек")
