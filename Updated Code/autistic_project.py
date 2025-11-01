# install in your pc: Python: Python 3.10.0 + Git: git version 2.51.0.windows.2
# python -m venv myenv

# myenv\Scripts\activate


# pip install torch torchvision pillow streamlit pytorch-grad-cam opencv-python pydicom tqdm

# pip install git+https://github.com/jacobgil/pytorch-grad-cam.git

# pip install -r requirements.txt


#  python .\autistic_project.py

# streamlit run .\autistic_app.py

import os
import re
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# -------- CONFIG ----------
DATA_ROOT = r"AUTISMDATASET"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VALID_DIR = os.path.join(DATA_ROOT, "valid")
TEST_DIR = os.path.join(DATA_ROOT, "test")
MODEL_PATH = "autism_cnn.pth"
NUM_CLASSES = 2
IMG_SIZE = 224
BATCH_SIZE = 8        # keep small due to 2GB VRAM
NUM_EPOCHS = 20
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
# --------------------------

CLASS_MAPPING = {"Autistic": 1, "Non_Autistic": 0, "Non-Autistic": 0, "NonCancer": 0}  # flexible mapping

# Utility to guess label from filename, e.g. "Autistic.0.jpg" or folder name.
def infer_label_from_path(p: Path) -> Optional[int]:
    # check parent folder names
    parent = p.parent.name
    if parent in CLASS_MAPPING:
        return CLASS_MAPPING[parent]
    # check filename
    name = p.name
    match = re.match(r"(Autistic|Non_Autistic|Non-Autistic|NonCancer|NonCancer)\b", name, flags=re.IGNORECASE)
    if match:
        lab = match.group(1)
        # normalize
        key = lab if lab in CLASS_MAPPING else lab.replace('-', '_')
        return CLASS_MAPPING.get(key, None)
    # fallback: try keywords anywhere
    if "autistic" in name.lower():
        return 1
    if "non" in name.lower():
        return 0
    return None

class FlexibleImageDataset(Dataset):
    """
    Works with:
    - root/class_name/*.jpg  (standard)
    - root/*.jpg where filenames indicate class: Autistic.0.jpg
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.items = []
        # collect images recursively (one level deep is fine)
        for p in self.root_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                lbl = infer_label_from_path(p)
                if lbl is None:
                    # try parent-of-parent (in case they used consolidated folder)
                    lbl = infer_label_from_path(p.parent)
                if lbl is None:
                    # skip ambiguous files
                    continue
                self.items.append((str(p), lbl))
        if len(self.items) == 0:
            raise RuntimeError(f"No images found under {root_dir} (or labels couldn't be inferred).")
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

# Small memory-friendly CNN (lightweight)
class SmallCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_transforms(train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

def train():
    train_ds = FlexibleImageDataset(TRAIN_DIR, transform=get_transforms(True))
    val_ds = FlexibleImageDataset(VALID_DIR, transform=get_transforms(False))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = SmallCNN(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)  # Removed verbose=True

    best_val_loss = 1e9
    for epoch in range(1, NUM_EPOCHS+1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [train]")
        for imgs, labels in loop:
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
            loop.set_postfix(loss=running_loss/total, acc=correct/total)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)
        val_loss /= val_total
        val_acc = val_correct / val_total
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Saved best model to {MODEL_PATH}")

if __name__ == "__main__":
    train()
