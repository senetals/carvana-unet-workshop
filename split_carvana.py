# split_carvana.py
import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# === CONFIG ===
BASE_DIR = Path("~/my_py_files/carvana").expanduser()
RAW_IMG_DIR = BASE_DIR / "raw" / "train"
RAW_MASK_DIR = BASE_DIR / "raw" / "train_masks"
SPLIT_DIR = BASE_DIR / "split"

TRAIN_RATIO = 0.8
RANDOM_SEED = 42
# =============

def main():
    print("Carvana Train/Val Split")
    print(f"Raw images: {RAW_IMG_DIR}")
    print(f"Raw masks:  {RAW_MASK_DIR}")
    print(f"Output:     {SPLIT_DIR}\n")

    # Validate raw data
    if not RAW_IMG_DIR.exists():
        print(f"ERROR: {RAW_IMG_DIR} not found!")
        print("Run download commands first.")
        return
    if not RAW_MASK_DIR.exists():
        print(f"ERROR: {RAW_MASK_DIR} not found!")
        return

    # Create output dirs
    for split in ["train", "val"]:
        for typ in ["images", "masks"]:
            (SPLIT_DIR / split / typ).mkdir(parents=True, exist_ok=True)

    # Get all car IDs from image filenames
    img_files = [f for f in RAW_IMG_DIR.iterdir() if f.suffix == ".jpg"]
    car_ids = sorted({f.stem.split('_')[0] for f in img_files})

    print(f"Found {len(car_ids)} unique cars (16 angles each)")

    # Split by car ID
    random.seed(RANDOM_SEED)
    random.shuffle(car_ids)
    split_idx = int(TRAIN_RATIO * len(car_ids))
    train_ids = set(car_ids[:split_idx])
    val_ids = set(car_ids[split_idx:])

    print(f"Train cars: {len(train_ids)} | Val cars: {len(val_ids)}")

    # Copy with progress
    def copy_files(car_set, split_name):
        img_out = SPLIT_DIR / split_name / "images"
        mask_out = SPLIT_DIR / split_name / "masks"
        count = 0
        for car_id in tqdm(car_set, desc=f"Copying {split_name}"):
            for angle in range(1, 17):
                img_name = f"{car_id}_{angle:02d}.jpg"
                mask_name = f"{car_id}_{angle:02d}_mask.gif"
                img_src = RAW_IMG_DIR / img_name
                mask_src = RAW_MASK_DIR / mask_name
                if img_src.exists() and mask_src.exists():
                    shutil.copy(img_src, img_out / img_name)
                    shutil.copy(mask_src, mask_out / mask_name)
                    count += 1
        print(f"{split_name.upper()}: {count} images copied")

    copy_files(train_ids, "train")
    copy_files(val_ids, "val")

    print("\nSPLIT COMPLETE!")
    print(f"Train: {SPLIT_DIR}/train/images")
    print(f"  Val: {SPLIT_DIR}/val/images")

if __name__ == "__main__":
    main()