# EfficientNet.py
import os, re
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# -------- CONFIG ----------
DATA_ROOT = r"AUTISMDATASET"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VALID_DIR = os.path.join(DATA_ROOT, "valid")
MODEL_PATH = "EfficientNet.pth"
NUM_CLASSES = 2
IMG_SIZE = 224
BATCH_SIZE = 8
NUM_EPOCHS = 20
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
# --------------------------

CLASS_MAPPING = {"Autistic": 1, "Non_Autistic": 0, "Non-Autistic": 0, "NonCancer": 0}

def infer_label_from_path(p: Path):
    parent = p.parent.name
    if parent in CLASS_MAPPING:
        return CLASS_MAPPING[parent]
    name = p.name
    match = re.match(r"(Autistic|Non_Autistic|Non-Autistic|NonCancer)", name, flags=re.IGNORECASE)
    if match:
        key = match.group(1).replace('-', '_')
        return CLASS_MAPPING.get(key, None)
    if "autistic" in name.lower():
        return 1
    if "non" in name.lower():
        return 0
    return None

class FlexibleImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.items = []
        for p in self.root_dir.rglob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                lbl = infer_label_from_path(p)
                if lbl is not None:
                    self.items.append((str(p), lbl))
        if len(self.items) == 0:
            raise RuntimeError(f"No images found in {root_dir}")

    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def train():
    train_ds = FlexibleImageDataset(TRAIN_DIR, transform=get_transforms(True))
    val_ds = FlexibleImageDataset(VALID_DIR, transform=get_transforms(False))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val = float("inf")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in tqdm(train_loader, desc=f"EfficientNet Epoch {epoch+1}/{NUM_EPOCHS}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
        acc = correct / total
        print(f"Train Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.4f}")

        # Validation
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                loss = criterion(out, labels)
                val_loss += loss.item()
                val_correct += (out.argmax(1) == labels).sum().item()
        val_loss /= len(val_loader)
        val_acc = val_correct / len(val_ds)
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"✅ Saved best model to {MODEL_PATH}")

if __name__ == "__main__":
    train()
