# predict.py — RANDOM IMAGE INFERENCE + VISUALIZATION (OPTIMIZED)

import os
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# ====================== MODEL DEFINITION (UNCHANGED) ======================
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): 
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_c=3, out_c=1):
        super().__init__()
        self.d1 = ConvBlock(in_c, 64); self.p1 = nn.MaxPool2d(2)
        self.d2 = ConvBlock(64, 128); self.p2 = nn.MaxPool2d(2)
        self.d3 = ConvBlock(128, 256); self.p3 = nn.MaxPool2d(2)
        self.d4 = ConvBlock(256, 512); self.p4 = nn.MaxPool2d(2)
        self.b  = ConvBlock(512, 1024)
        self.u1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.c1 = ConvBlock(1024, 512)
        self.u2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.c2 = ConvBlock(512, 256)
        self.u3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.c3 = ConvBlock(256, 128)
        self.u4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.c4 = ConvBlock(128, 64)
        self.out = nn.Conv2d(64, out_c, 1)

    def forward(self, x):
        d1 = self.d1(x); p1 = self.p1(d1)
        d2 = self.d2(p1); p2 = self.p2(d2)
        d3 = self.d3(p2); p3 = self.p3(d3)
        d4 = self.d4(p3); p4 = self.p4(d4)
        b = self.b(p4)
        u1 = self.u1(b); u1 = torch.cat([u1, d4], 1); u1 = self.c1(u1)
        u2 = self.u2(u1); u2 = torch.cat([u2, d3], 1); u2 = self.c2(u2)
        u3 = self.u3(u2); u3 = torch.cat([u3, d2], 1); u3 = self.c3(u3)
        u4 = self.u4(u3); u4 = torch.cat([u4, d1], 1); u4 = self.c4(u4)
        return self.out(u4)

# ====================== DEVICE & MODEL ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = UNet().to(device)
state_dict = torch.load("unet_small_final.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

print("Model loaded successfully.")

# ====================== RANDOM IMAGE SELECTION ======================
IMG_DIR = "carvana_small/train/images"
MASK_DIR = "carvana_small/train/masks"

img_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")])
assert len(img_files) > 0, "No images found in carvana/raw/train"

img_name = random.choice(img_files)
img_path = os.path.join(IMG_DIR, img_name)
mask_path = os.path.join(
    MASK_DIR,
    img_name.replace(".jpg", "_mask.gif")
)

print(f"Randomly selected image: {img_name}")

# ====================== TRANSFORMS ======================
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ====================== LOAD IMAGE & MASK ======================
img_pil = Image.open(img_path).convert("RGB")
mask_pil = Image.open(mask_path).convert("L")

img_tensor = transform(img_pil).unsqueeze(0).to(device)

# ====================== PREDICTION (OPTIMIZED) ======================
with torch.no_grad():
    logits = model(img_tensor)
    prob_map = torch.sigmoid(logits)[0, 0]
    pred_mask = (prob_map > 0.5).float()

prob_map = prob_map.cpu()
pred_mask = pred_mask.cpu()

# ====================== DENORMALIZE IMAGE ======================
mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

img_show = img_tensor[0].cpu() * std + mean
img_show = img_show.clamp(0, 1)

mask_show = mask_pil.resize((512, 512), Image.NEAREST)

# ====================== VISUALIZATION ======================
plt.figure(figsize=(20, 5))

plt.subplot(1, 4, 1)
plt.imshow(img_show.permute(1,2,0))
plt.title("Input Image")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(mask_show, cmap="gray")
plt.title("Ground Truth Mask")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(pred_mask, cmap="gray")
plt.title("Predicted Mask")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(prob_map, cmap="hot", vmin=0, vmax=1)
plt.title("Prediction Heatmap")
plt.colorbar(fraction=0.046)
plt.axis("off")

plt.tight_layout()

# ====================== SAVE OUTPUT ======================
out_name = f"prediction_{img_name.replace('.jpg','')}.png"
plt.savefig(out_name, dpi=200, bbox_inches="tight")
plt.show()

print(f"SAVED: {out_name}")

# ====================== CLEANUP ======================
del img_tensor, logits
torch.cuda.empty_cache()
