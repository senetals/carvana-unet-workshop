import shutil
import random
from pathlib import Path

RAW_IMG_DIR = Path("carvana/raw/images")
RAW_MASK_DIR = Path("carvana/raw/masks")

OUT_TRAIN_IMG = Path("carvana_small/train/images")
OUT_TRAIN_MASK = Path("carvana_small/train/masks")
OUT_VAL_IMG = Path("carvana_small/val/images")
OUT_VAL_MASK = Path("carvana_small/val/masks")

NUM_SAMPLES = 96
VAL_RATIO = 0.2

def prepare_small_dataset():
    images = sorted(RAW_IMG_DIR.glob("*.jpg"))
    random.shuffle(images)

    selected = images[:NUM_SAMPLES]
    split = int(len(selected) * (1 - VAL_RATIO))

    train_imgs = selected[:split]
    val_imgs = selected[split:]

    for d in [OUT_TRAIN_IMG, OUT_TRAIN_MASK, OUT_VAL_IMG, OUT_VAL_MASK]:
        d.mkdir(parents=True, exist_ok=True)

    def copy(img_list, img_out, mask_out):
        for img in img_list:
            mask = RAW_MASK_DIR / img.name.replace(".jpg", "_mask.gif")
            shutil.copy(img, img_out / img.name)
            shutil.copy(mask, mask_out / mask.name)

    copy(train_imgs, OUT_TRAIN_IMG, OUT_TRAIN_MASK)
    copy(val_imgs, OUT_VAL_IMG, OUT_VAL_MASK)

    print("Small dataset created successfully")

if __name__ == "__main__":
    prepare_small_dataset()
