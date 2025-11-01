# DenseNet.py
import os, re
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

DATA_ROOT = r"AUTISMDATASET"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VALID_DIR = os.path.join(DATA_ROOT, "valid")
MODEL_PATH = "DenseNet.pth"
NUM_CLASSES = 2
IMG_SIZE = 224
BATCH_SIZE = 8
NUM_EPOCHS = 20
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def infer_label_from_path(p: Path):
    if "autistic" in p.name.lower() or "autistic" in p.parent.name.lower():
        return 1
    if "non" in p.name.lower() or "non" in p.parent.name.lower():
        return 0
    return None

class FlexibleImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root = Path(root_dir)
        self.tr = transform
        self.items = [(str(p), infer_label_from_path(p)) for p in self.root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and infer_label_from_path(p) is not None]
        if not self.items: raise RuntimeError("No images found.")
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, l = self.items[i]
        img = Image.open(p).convert("RGB")
        return (self.tr(img) if self.tr else img), torch.tensor(l)

def get_transforms(train=True):
    t = [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
         transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]
    if train: t.insert(1, transforms.RandomHorizontalFlip())
    return transforms.Compose(t)

def train():
    train_loader = DataLoader(FlexibleImageDataset(TRAIN_DIR, get_transforms(True)), BATCH_SIZE, True)
    val_loader = DataLoader(FlexibleImageDataset(VALID_DIR, get_transforms(False)), BATCH_SIZE, False)

    model = models.densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val = float("inf")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = correct = total = 0
        for imgs, labels in tqdm(train_loader, desc=f"DenseNet Epoch {epoch+1}/{NUM_EPOCHS}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
        print(f"Train Loss={total_loss/len(train_loader):.4f}, Acc={correct/total:.4f}")

        model.eval()
        val_loss = val_correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                loss = criterion(out, labels)
                val_loss += loss.item()
                val_correct += (out.argmax(1) == labels).sum().item()
        val_loss /= len(val_loader)
        val_acc = val_correct / len(val_loader.dataset)
        print(f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"✅ Saved best model to {MODEL_PATH}")

if __name__ == "__main__":
    train()
