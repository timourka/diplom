from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")  # базовая предобученная модель

    model.train(
        data="expiry_data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        name="expiry_all"
    )

if __name__ == "__main__":
    main()
