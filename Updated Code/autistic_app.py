# ===========================
# autistic_app.py
# Streamlit app for Autism Detection using multiple models
# ===========================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import streamlit as st
import torchvision.transforms as transforms
import cv2
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import models

# ------------------------
# CONFIG
# ------------------------
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.set_page_config(page_title="Autism Detection App", layout="wide")
st.title("🧠 Autism Detection using Deep Learning")

# ------------------------
# MODEL DEFINITIONS
# ------------------------

# SmallCNN same as autistic_project.py
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
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# EfficientNet Model
def build_efficientnet(num_classes=2):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

# MobileNet Model
def build_mobilenet(num_classes=2):
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model

# DenseNet Model
def build_densenet(num_classes=2):
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model

# ResNet Model
def build_resnet(num_classes=2):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ------------------------
# UTILS
# ------------------------

@st.cache_resource
def load_model(model_name):
    """Load model dynamically based on sidebar selection."""
    model_path = f"{model_name}.pth"
    if not os.path.exists(model_path):
        st.error(f"Model weights not found: {model_path}")
        return None

    if model_name == "autism_cnn":
        model = SmallCNN(num_classes=2)
        target_layer = model.features[-2]

    elif model_name == "EfficientNet":
        model = build_efficientnet(num_classes=2)
        target_layer = model.features[-1]

    elif model_name == "MobileNet":
        model = build_mobilenet(num_classes=2)
        target_layer = model.features[-1]

    elif model_name == "DenseNet":
        model = build_densenet(num_classes=2)
        target_layer = model.features[-1]  # last conv layer

    elif model_name == "ResNet":
        model = build_resnet(num_classes=2)
        target_layer = model.layer4[-1]  # last block conv layer

    else:
        st.error("Unknown model selected.")
        return None

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model, target_layer


def preprocess_image(pil_image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return transform(pil_image).unsqueeze(0)


def predict_and_cam(model, target_layer, pil_image, cam_type="GradCAM"):
    """Run prediction + Grad-CAM visualization."""
    input_tensor = preprocess_image(pil_image).to(DEVICE)
    img_np = np.array(pil_image.resize((IMG_SIZE, IMG_SIZE))) / 255.0
    img_np = np.float32(img_np)

    # Prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
        pred_class = "Autistic" if pred.item() == 1 else "Non-Autistic"

    # Grad-CAM
    cam_algo = {
        "GradCAM": GradCAM,
        "GradCAM++": GradCAMPlusPlus,
        "EigenCAM": EigenCAM
    }.get(cam_type, GradCAM)

    cam = cam_algo(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor)[0, :]
    grayscale_cam = cv2.resize(grayscale_cam, (IMG_SIZE, IMG_SIZE))
    cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    return pred_class, conf.item(), cam_image


# ------------------------
# SIDEBAR
# ------------------------
st.sidebar.header("🔍 Model Selection & Options")
model_choice = st.sidebar.selectbox(
    "Choose a model:",
    ["autism_cnn", "EfficientNet", "MobileNet", "DenseNet", "ResNet"]
)
cam_choice = st.sidebar.radio("Choose CAM type:", ["GradCAM", "GradCAM++", "EigenCAM"])

model_info = {
    "autism_cnn": "A lightweight custom CNN built for low-resource autism image classification.",
    "EfficientNet": "EfficientNet-B0 pre-trained structure adapted for binary classification.",
    "MobileNet": "MobileNetV3-Small optimized for mobile and low-memory environments.",
    "DenseNet": "DenseNet-121 with dense connectivity for better feature reuse.",
    "ResNet": "ResNet-18 with residual skip connections for stable deep learning."
}

st.sidebar.markdown(f"**Model Info:** {model_info[model_choice]}")


# ------------------------
# MAIN CONTENT
# ------------------------
st.markdown("### 🧩 Upload an image to analyze")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    st.image(pil_img, caption="Uploaded Image", use_container_width=True)

    if st.button("Run Prediction"):
        st.write("Running prediction...")

        model, target_layer = load_model(model_choice)
        if model is not None:
            pred, conf, cam_img = predict_and_cam(model, target_layer, pil_img, cam_choice)

            st.success(f"**Prediction:** {pred} ({conf*100:.2f}% confidence)")

            st.markdown("#### 🔥 Grad-CAM Visualization")
            st.image(cam_img, use_container_width=True)
