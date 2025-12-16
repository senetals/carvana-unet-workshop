# unet_speed_new.py
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
#from new_dataset import train_loader, val_loader
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import CarvanaDataset
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

#torch.set_num_threads(8)
#print("Training on SMALL DATASET | 8 threads | batch=4")
torch.set_num_threads(min(2, os.cpu_count()))

# ====================== DATASETS ======================
train_dataset = CarvanaDataset(
    "carvana_small/train/images",
    "carvana_small/train/masks"
)
val_dataset = CarvanaDataset(
    "carvana_small/val/images",
    "carvana_small/val/masks"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=1,    # ← CHANGE THIS
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,     # ← FIXED (was train_dataset)
    batch_size=1,    # ← CHANGE THIS
    shuffle=False,
    num_workers=2
)

# ====================== MODEL ======================
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

# ====================== TRAINING SETUP ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = UNet().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

#model = model.to(device)
#criterion = criterion.to(device)

scaler = GradScaler(enabled=(device.type == "cuda"))

# ====================== TRAIN LOOP ======================
for epoch in range(1, 6):
    model.train()
    total_loss = 0.0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=(device.type == "cuda")):
            outputs = model(x)          # ← FIXED variable name
            loss = criterion(outputs, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f}")

    torch.cuda.empty_cache()

# ====================== DENORMALIZE & VISUALIZE ======================
model.eval()
x, y = next(iter(val_loader))
x = x.to(device)
y = y.to(device)

with torch.no_grad():
    pred = torch.sigmoid(model(x))

# Move to CPU for plotting
x_cpu = x.cpu()
y_cpu = y.cpu()
pred_cpu = pred.cpu()

# Denormalize input image (REVERSE of training transform)
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
x_show = x_cpu * std + mean  # ← CORRECT ORDER
x_show = x_show.clamp(0, 1)  # ← CRITICAL: prevent negative/overflow

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(x_show[0].permute(1, 2, 0).numpy())
axes[0].set_title("Input Image")
axes[0].axis('off')

axes[1].imshow(y_cpu[0].squeeze().numpy(), cmap='gray')
axes[1].set_title("Ground Truth")
axes[1].axis('off')

axes[2].imshow((pred_cpu[0].squeeze() > 0.5).numpy(), cmap='gray')
axes[2].set_title("Prediction (threshold 0.5)")
axes[2].axis('off')

plt.tight_layout()

# SAVE BEFORE SHOW (important!)
plt.savefig("unet_first_prediction.png", dpi=150, bbox_inches='tight')
print("Prediction image saved: unet_first_prediction.png")

plt.show()

# ====================== MODEL SAVE & VERIFY ======================
save_dir = os.getcwd()
print(f"\nSaving model to: {save_dir}")

main_path = os.path.join(save_dir, "unet_small_final.pth")
torch.save(model.state_dict(), main_path)
print(f"Model saved as: {main_path}")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_path = os.path.join(save_dir, f"unet_small_{timestamp}.pth")
torch.save(model.state_dict(), backup_path)
print(f"Backup saved as: {backup_path}")

# Verification
print("\nVerifying saved files exist...")
assert os.path.exists(main_path), f"FAILED: {main_path} not created!"
assert os.path.exists(backup_path), f"FAILED: {backup_path} not created!"
print("Both .pth files saved and verified!")
