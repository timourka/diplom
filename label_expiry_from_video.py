import cv2
import os
import argparse

# Глобальные переменные для callback'а мыши
drawing = False
start_disp = None         # старт прямоугольника в координатах отображаемого окна
end_disp = None           # конец прямоугольника (window coords)
box_ready = False
final_box_orig = None     # финальный бокс в координатах оригинального кадра

current_frame_orig = None  # оригинальный кадр (полный размер)
display_base = None        # уменьшенный кадр без разметки
display_frame = None       # уменьшенный кадр с текущим прямоугольником

frame_w = None
frame_h = None
disp_scale = 1.0           # масштаб отображения (<= 1.0)

# Ограничение окна (можешь поменять под свой монитор)
MAX_WIN_W = 1280
MAX_WIN_H = 720


def mouse_callback(event, x, y, flags, param):
    """
    Обработка мыши на уменьшенном изображении.
    Мы рисуем bbox в координатах окна (display),
    а при завершении пересчитываем в оригинальные координаты кадра.
    """
    global drawing, start_disp, end_disp, box_ready, final_box_orig, display_frame, display_base, disp_scale

    if display_base is None:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_disp = (x, y)
        end_disp = (x, y)
        box_ready = False
        final_box_orig = None

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        end_disp = (x, y)
        display_frame = display_base.copy()
        x1, y1 = start_disp
        x2, y2 = end_disp
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_disp = (x, y)
        x1, y1 = start_disp
        x2, y2 = end_disp

        # нормализуем порядок углов
        x_min_disp, x_max_disp = sorted([x1, x2])
        y_min_disp, y_max_disp = sorted([y1, y2])

        # проверка на минимальный размер (в окне)
        if x_max_disp - x_min_disp < 5 or y_max_disp - y_min_disp < 5:
            print("Слишком маленький прямоугольник, не сохраняю.")
            return

        # рисуем финальный прямоугольник в окне
        display_frame = display_base.copy()
        cv2.rectangle(display_frame, (x_min_disp, y_min_disp),
                      (x_max_disp, y_max_disp), (0, 255, 0), 2)

        # пересчёт в координаты оригинального кадра
        # x_orig = x_disp / disp_scale
        x_min_orig = int(x_min_disp / disp_scale)
        y_min_orig = int(y_min_disp / disp_scale)
        x_max_orig = int(x_max_disp / disp_scale)
        y_max_orig = int(y_max_disp / disp_scale)

        final_box_orig = (x_min_orig, y_min_orig, x_max_orig, y_max_orig)
        box_ready = True


def save_yolo_annotation(out_img_dir, out_lbl_dir, frame_idx, frame, bbox):
    """
    Сохранить кадр + разметку в YOLO-формате.
    bbox: (x_min, y_min, x_max, y_max) в пикселях ОРИГИНАЛЬНОГО кадра.
    """
    global frame_w, frame_h

    x_min, y_min, x_max, y_max = bbox

    img_name = f"frame_{frame_idx:06d}.jpg"
    lbl_name = f"frame_{frame_idx:06d}.txt"

    img_path = os.path.join(out_img_dir, img_name)
    lbl_path = os.path.join(out_lbl_dir, lbl_name)

    # сохраняем оригинальный кадр
    cv2.imwrite(img_path, frame)

    # YOLO: class_id cx cy w h (нормированные 0..1)
    cx = (x_min + x_max) / 2.0 / frame_w
    cy = (y_min + y_max) / 2.0 / frame_h
    bw = (x_max - x_min) / frame_w
    bh = (y_max - y_min) / frame_h

    class_id = 0  # единственный класс: expiry_date
    line = f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"

    with open(lbl_path, "w", encoding="utf-8") as f:
        f.write(line)

    print(f"  ✔ Сохранил разметку для кадра {frame_idx} -> {img_name}")


def main():
    parser = argparse.ArgumentParser(description="Разметка срока годности из видео (с масштабированием)")
    parser.add_argument("--video", required=True, help="Путь к видеофайлу (.mp4/.mov и т.п.)")
    parser.add_argument("--out", required=True, help="Папка для датасета (будут subdirs images/, labels/)")
    parser.add_argument("--step", type=int, default=1,
                        help="Брать каждый N-й кадр (по умолчанию 1 — каждый)")
    args = parser.parse_args()

    video_path = args.video
    out_root = args.out
    step = max(1, args.step)

    out_img_dir = os.path.join(out_root, "images")
    out_lbl_dir = os.path.join(out_root, "labels")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Не удалось открыть видео: {video_path}")
        return

    global current_frame_orig, display_base, display_frame, frame_w, frame_h
    global disp_scale, box_ready, final_box_orig

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Видео: {video_path}")
    print(f"  Кадров: {frame_count}, размер: {frame_w}x{frame_h}")
    print("Управление:")
    print("  ЛКМ + перетаскивание — выделить дату (бокс сохранится, сразу следующий кадр)")
    print("  s — пропустить кадр без разметки")
    print("  q — выйти\n")

    cv2.namedWindow("label", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("label", mouse_callback)

    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Видео закончилось.")
            break

        # пропускаем кадры по шагу
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        current_frame_orig = frame.copy()

        # считаем масштаб для влезания в окно
        h0, w0 = current_frame_orig.shape[:2]
        sx = MAX_WIN_W / float(w0)
        sy = MAX_WIN_H / float(h0)
        disp_scale = min(sx, sy, 1.0)  # только уменьшение, не увеличиваем

        if disp_scale < 1.0:
            new_w = int(w0 * disp_scale)
            new_h = int(h0 * disp_scale)
            display_base = cv2.resize(current_frame_orig, (new_w, new_h))
        else:
            display_base = current_frame_orig.copy()

        display_frame = display_base.copy()
        box_ready = False
        final_box_orig = None

        while True:
            vis = display_frame.copy()
            cv2.putText(
                vis,
                "Draw box with mouse. 's' - skip, 'q' - quit",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )
            cv2.putText(
                vis,
                f"Frame {frame_idx}/{frame_count}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

            cv2.imshow("label", vis)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('q'):
                print("Выход по 'q'.")
                cap.release()
                cv2.destroyAllWindows()
                print(f"Сохранено кадров: {saved}")
                return

            if key == ord('s'):
                print(f"  ↷ Пропустил кадр {frame_idx}")
                break

            if box_ready and final_box_orig is not None:
                # сохраняем разметку в оригинальных координатах
                save_yolo_annotation(out_img_dir, out_lbl_dir, frame_idx,
                                     current_frame_orig, final_box_orig)
                saved += 1
                break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nГотово. Сохранено размеченных кадров: {saved}")


if __name__ == "__main__":
    main()
