import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
import numpy as np

class CarvanaDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=256):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)

        self.images = sorted(self.img_dir.glob("*.jpg"))
        if len(self.images) == 0:
            raise RuntimeError(f"No images found in {self.img_dir}")

        self.transform = T.Compose([
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Use NEAREST for masks to keep binary crisp!
        self.mask_transform = T.Compose([
            T.Resize((size, size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.mask_dir / img_path.name.replace(".jpg", "_mask.gif")

        img = Image.open(img_path).convert("RGB")

        # Load mask as PIL Image first
        mask = Image.open(mask_path).convert("L")

        # Optional seed for synchronized transforms (if you add random ones later)
        # But for now, deterministic is fine

        img = self.transform(img)  # This includes ToTensor + Normalize → fine for images

        # Apply mask transform on PIL Image
        mask = self.mask_transform(mask)  # Now Resize on PIL → ToTensor

        # Make mask strictly binary (0 or 1) after ToTensor
        # ToTensor scales 0-255 → 0.0-1.0, so >0.5 works
        mask = (mask > 0.5).float()  # → [1, H, W] with 0.0 or 1.0

        return img, mask