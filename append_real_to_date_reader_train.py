import csv
import shutil
from pathlib import Path

real_dir = Path("D:/diplom/date_reader_real")
work_dir = Path("D:/diplom/date_reader_work_v5_gpu")
repeat = 40

src_csv = real_dir / "labels.csv"
dst_img_dir = work_dir / "crops" / "train"
dst_csv = work_dir / "labels_train.csv"

dst_img_dir.mkdir(parents=True, exist_ok=True)

def norm_text(s: str) -> str:
    s = (s or "").strip().upper()
    s = s.replace("\\", "/").replace("|", "/")
    s = s.replace(" ", ".")
    allowed = set("0123456789./-")
    return "".join(ch for ch in s if ch in allowed)

rows_to_add = []

with src_csv.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    for i, r in enumerate(reader, start=1):
        rel_img = r.get("image") or r.get("path")
        text = norm_text(r.get("text") or r.get("label") or "")
        if not rel_img or not text:
            continue

        src_img = real_dir / rel_img
        if not src_img.exists():
            print("[SKIP] not found:", src_img)
            continue

        ext = src_img.suffix.lower() or ".jpg"
        dst_name = f"myreal_{i:05d}{ext}"
        dst_img = dst_img_dir / dst_name
        shutil.copy2(src_img, dst_img)

        rel_for_csv = f"crops/train/{dst_name}"
        for _ in range(repeat):
            rows_to_add.append([rel_for_csv, text])

with dst_csv.open("a", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows_to_add)

print(f"[OK] added rows: {len(rows_to_add)}")
print(f"[OK] real images copied to: {dst_img_dir}")
print(f"[OK] appended to: {dst_csv}")
