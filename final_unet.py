# full_train.py — FINAL FULL TRAINING SCRIPT (optimized for workshop)
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import CarvanaDataset
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# For reproducibility (great for demos!)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

print("5 EPOCHS ON FULL CARVANA DATASET — STARTING NOW!")
torch.set_num_threads(min(4, os.cpu_count()))  # Slight bump, safe

# ====================== DATASETS ======================
train_dataset = CarvanaDataset(
    "carvana/split/train/images",
    "carvana/split/train/masks"
)
val_dataset = CarvanaDataset(
    "carvana/split/val/images",
    "carvana/split/val/masks"
)

# Try batch_size=8 (GTX 1650 can handle 256x256 UNet easily—faster training!)
train_loader = DataLoader(
    train_dataset,
    batch_size=8,        
    shuffle=True,
    num_workers=4,       # More workers = faster loading
    pin_memory=True      # Speedup for GPU
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,        
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# === YOUR UNET (unchanged—solid classic!) ===
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
    def forward(self, x): return self.block(x)

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

# === TRAINING LOOP (added simple val loss for monitoring) ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = UNet().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-4)  # Good default

scaler = GradScaler(enabled=(device.type == "cuda"))

def freeze_bn(m):
    if isinstance(m, nn.BatchNorm2d):
        m.eval()

model.apply(freeze_bn)

for epoch in range(1, 6):
    model.train()
    model.apply(freeze_bn)
    epoch_loss = 0.0
    
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/5 Train"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=(device.type == "cuda")):
            pred = model(x)
            loss = criterion(pred, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / len(train_loader)
    print(f"\nEPOCH {epoch}/5 TRAIN — Avg Loss: {avg_train_loss:.5f}")

    # Quick val (helps spot issues early)
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Validating"):
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            val_loss += criterion(pred, y).item()
    avg_val_loss = val_loss / len(val_loader)
    print(f"EPOCH {epoch}/5 VAL   — Avg Loss: {avg_val_loss:.5f}")

    torch.cuda.empty_cache()

# ====================== SAVE & VISUALIZE (unchanged, solid) ======================
torch.save(model.state_dict(), "unet_full_5epochs.pth")
print("MODEL SAVED: unet_full_5epochs.pth")

model.eval()
x, y = next(iter(val_loader))
x = x.to(device)
y = y.to(device)

with torch.no_grad():
    pred = torch.sigmoid(model(x))

x_cpu = x.cpu()
y_cpu = y.cpu()
pred_cpu = pred.cpu()

mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
x_show = x_cpu * std + mean
x_show = x_show.clamp(0, 1)

plt.figure(figsize=(15,5))
plt.subplot(1,3,1); plt.imshow(x_show[0].permute(1,2,0)); plt.title("Input")
plt.subplot(1,3,2); plt.imshow(y_cpu[0].squeeze(), cmap='gray'); plt.title("Ground Truth")
plt.subplot(1,3,3); plt.imshow((pred_cpu[0].squeeze() > 0.5), cmap='gray'); plt.title("Prediction")
plt.tight_layout()
plt.savefig("unet_full_5epochs_result.png", dpi=200)
plt.show()

print("RESULT SAVED: unet_full_5epochs_result.png")