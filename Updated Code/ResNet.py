# ResNet.py
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
MODEL_PATH = "ResNet.pth"
NUM_CLASSES = 2
IMG_SIZE = 224
BATCH_SIZE = 8
NUM_EPOCHS = 20
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def infer_label_from_path(p: Path):
    n = p.parent.name.lower() + p.name.lower()
    if "autistic" in n: return 1
    if "non" in n: return 0
    return None

class FlexibleImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root); self.tr = transform
        self.items = [(str(p), infer_label_from_path(p)) for p in self.root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and infer_label_from_path(p) is not None]
        if not self.items: raise RuntimeError("No images found.")
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p,l=self.items[i]; img=Image.open(p).convert("RGB")
        return (self.tr(img) if self.tr else img), torch.tensor(l)

def get_transforms(train=True):
    t=[transforms.Resize((IMG_SIZE,IMG_SIZE)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]
    if train: t.insert(1,transforms.RandomHorizontalFlip())
    return transforms.Compose(t)

def train():
    train_dl=DataLoader(FlexibleImageDataset(TRAIN_DIR,get_transforms(True)),BATCH_SIZE,True)
    val_dl=DataLoader(FlexibleImageDataset(VALID_DIR,get_transforms(False)),BATCH_SIZE,False)
    model=models.resnet18(pretrained=True); model.fc=nn.Linear(model.fc.in_features,NUM_CLASSES); model=model.to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=LR); crit=nn.CrossEntropyLoss(); best=float("inf")
    for ep in range(NUM_EPOCHS):
        model.train(); loss_sum=correct=total=0
        for x,y in tqdm(train_dl,desc=f"ResNet Epoch {ep+1}/{NUM_EPOCHS}"):
            x,y=x.to(DEVICE),y.to(DEVICE); opt.zero_grad(); o=model(x); l=crit(o,y); l.backward(); opt.step()
            loss_sum+=l.item(); correct+=(o.argmax(1)==y).sum().item(); total+=len(y)
        print(f"Train Loss={loss_sum/len(train_dl):.4f}, Acc={correct/total:.4f}")
        model.eval(); vloss=vcorr=0
        with torch.no_grad():
            for x,y in val_dl:
                x,y=x.to(DEVICE),y.to(DEVICE); o=model(x); l=crit(o,y)
                vloss+=l.item(); vcorr+=(o.argmax(1)==y).sum().item()
        vloss/=len(val_dl); vacc=vcorr/len(val_dl.dataset)
        print(f"Val Loss={vloss:.4f}, Acc={vacc:.4f}")
        if vloss<best: best=vloss; torch.save(model.state_dict(),MODEL_PATH); print(f"✅ Saved best model to {MODEL_PATH}")

if __name__=="__main__": train()
