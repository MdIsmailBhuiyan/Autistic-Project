# streamlit_app.py
import os
import io
from pathlib import Path

import streamlit as st
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import cv2

# grad-cam imports
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ---------------- CONFIG ----------------
MODEL_PATH = "autism_cnn.pth"   # path to saved model from train_cnn.py
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
# ----------------------------------------

CLASS_LABELS = {0: "Non_Autistic", 1: "Autistic"}

# Recreate same model class as training
class SmallCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
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

# Load model
@st.cache_resource
def load_model(path=MODEL_PATH):
    model = SmallCNN(num_classes=2)
    if not Path(path).exists():
        st.error(f"Model file not found at {path}. Train the model first (train_cnn.py) and place the file here.")
        return None
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# Transforms (must match training)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def pil_to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")

def prepare_tensor_from_pil(image: Image.Image):
    img = pil_to_rgb(image)
    t = transform(img).unsqueeze(0).to(DEVICE)
    return t

def predict_and_cam(pil_image: Image.Image, cam_type="gradcam"):
    if model is None:
        st.error("Model not loaded.")
        return None, None, None

    input_tensor = prepare_tensor_from_pil(pil_image)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(probs.argmax(dim=1).item())
        confidence = float(probs[0, pred_idx].cpu().numpy())

    # prepare image for cam: normalized RGB in [0,1]
    img_np = np.array(pil_image.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0

    # target layer for CAM
    target_layer = model.features[-2] if isinstance(model.features, nn.Sequential) else model.features

    # select CAM method
    cam_method = {
        "gradcam": GradCAM,
        "gradcam++": GradCAMPlusPlus,
        "eigen": EigenCAM
    }.get(cam_type, GradCAM)

    # create CAM instance (no use_cuda anymore)
    cam = cam_method(model=model, target_layers=[target_layer])
    if DEVICE.type == "cuda":
        cam = cam.cuda()

    # compute CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
    cam_resized = cv2.resize(grayscale_cam, (pil_image.width, pil_image.height))
    cam_on_image = show_cam_on_image(np.array(pil_image).astype(np.float32)/255.0, cam_resized, use_rgb=True)

    return CLASS_LABELS[pred_idx], confidence, cam_on_image

# Streamlit UI
st.set_page_config(page_title="Autism Detection (Explainable)", layout="centered")
st.title("Autism Detection (Explainable AI)")
st.write("Upload an image and view prediction + Grad-CAM explanations.")

cam_choice = st.sidebar.selectbox("CAM type", ["gradcam", "gradcam++", "eigen"])
uploaded = st.file_uploader("Upload image (JPG/PNG) or DICOM (.dcm)", type=["jpg","jpeg","png","dcm","tiff","tif","bmp"])

if uploaded is not None:
    bytes_data = uploaded.read()
    # handle DICOM if needed
    if uploaded.type == "application/dicom" or str(uploaded.name).lower().endswith(".dcm"):
        try:
            import pydicom
            dicom = pydicom.dcmread(io.BytesIO(bytes_data))
            arr = dicom.pixel_array.astype(np.float32)
            arr = (arr - arr.min())/(arr.max()-arr.min()+1e-8)
            # convert to 3-channel RGB
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            pil_img = Image.fromarray((arr*255).astype(np.uint8))
        except Exception as e:
            st.error(f"Failed to read DICOM: {e}")
            pil_img = None
    else:
        pil_img = Image.open(io.BytesIO(bytes_data)).convert("RGB")

    if pil_img is not None:
        
        st.image(pil_img, caption="Uploaded Image", use_container_width=True)
        st.write("Running prediction and CAM (this may take a moment)...")
        pred_label, conf, cam_img = predict_and_cam(pil_img, cam_choice)
        if pred_label is not None:
            st.subheader(f"Prediction: **{pred_label}**")
            st.write(f"Confidence: **{conf:.4f}**")
            st.image(cam_img, caption=f"{cam_choice.upper()} visualization", use_container_width=True)

            # Also show all three CAMs side-by-side
            st.markdown("---")
            st.subheader("Compare CAM methods")
            col1, col2, col3 = st.columns(3)
            for col, method in zip((col1, col2, col3), ("gradcam", "gradcam++", "eigen")):
                _, _, imgm = predict_and_cam(pil_img, method)
                col.image(imgm, caption=method.upper())
else:
    st.info("Upload an image to run prediction.")

st.sidebar.markdown("**Model:**** SmallCNN (lightweight). Train with `train_cnn.py` and ensure the saved `autism_cnn.pth` is in the same folder.")
