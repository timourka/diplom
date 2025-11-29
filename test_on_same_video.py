import cv2
import time
from ultralytics import YOLO

# ==== НАСТРОЙКИ ====

MODEL_PATH = "runs/detect/expiry_all/weights/best.pt"  # путь к обученной модели
VIDEO_SOURCE = "medias/IMG_5068.MOV"  # или 0 для вебки: VIDEO_SOURCE = 0

CONF_THRES = 0.3           # порог уверенности
MAX_WIN_W = 1280           # максимальная ширина окна
MAX_WIN_H = 720            # максимальная высота окна
ZOOM_STEP = 0.1            # шаг зума по клавишам +/-


def main():
    # Загружаем модель
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"❌ Не удалось открыть источник видео: {VIDEO_SOURCE}")
        return

    cv2.namedWindow("YOLO expiry_date", cv2.WINDOW_NORMAL)

    # Первое чтение, чтобы понять размер
    ret, frame = cap.read()
    if not ret:
        print("Не удалось прочитать первый кадр.")
        cap.release()
        return

    h0, w0 = frame.shape[:2]
    # базовый масштаб, чтобы влезть в окно
    base_sx = MAX_WIN_W / float(w0)
    base_sy = MAX_WIN_H / float(h0)
    base_scale = min(base_sx, base_sy, 1.0)

    zoom_factor = 1.0  # можно будет менять с клавиатуры

    # Вернёмся в начало видео
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    prev_time = time.time()
    fps = 0.0

    print("Управление:")
    print("  q  — выход")
    print("  +/= — увеличить масштаб")
    print("  -/_ — уменьшить масштаб")

    while True:
        ret, frame = cap.read()
        if not ret:
            # для файла — просто выходим; для вебки — можно сделать continue
            print("Видео закончилось или поток недоступен.")
            break

        t0 = time.time()

        # предсказание на одном кадре
        # model(...) возвращает список результатов; .plot() рисует боксы
        results = model(frame, conf=CONF_THRES, verbose=False)
        annotated = results[0].plot()  # BGR np.array

        # считаем FPS
        t1 = time.time()
        dt = t1 - prev_time
        prev_time = t1
        if dt > 0:
            fps = 1.0 / dt

        # накладываем текст FPS
        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # === МАСШТАБИРОВАНИЕ ДЛЯ ОТОБРАЖЕНИЯ ===
        # общий масштаб = базовый (влезть в окно) * zoom_factor
        disp_scale = base_scale * zoom_factor
        if disp_scale != 1.0:
            new_w = int(annotated.shape[1] * disp_scale)
            new_h = int(annotated.shape[0] * disp_scale)
            disp_frame = cv2.resize(annotated, (new_w, new_h))
        else:
            disp_frame = annotated

        cv2.imshow("YOLO expiry_date", disp_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key in (ord('+'), ord('=')):
            zoom_factor += ZOOM_STEP
            if zoom_factor > 3.0:
                zoom_factor = 3.0
            print(f"zoom_factor = {zoom_factor:.2f}")
        elif key in (ord('-'), ord('_')):
            zoom_factor -= ZOOM_STEP
            if zoom_factor < 0.2:
                zoom_factor = 0.2
            print(f"zoom_factor = {zoom_factor:.2f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
