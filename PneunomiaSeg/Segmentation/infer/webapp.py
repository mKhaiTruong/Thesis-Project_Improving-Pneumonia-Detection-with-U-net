import sys, io
import streamlit as st
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(r"..\PneunomiaSeg\SAM\CUS SAM")
from helpers import *
from dataset.unet_dataset import Test_UnetDataset

# ---------------------
# CONFIG
# ---------------------
model_dir = r"..\PneunomiaSeg\Segmentation\results\models"
exp_name  = "exp1"
img_pth   = [r"D:\Downloads\COVID-19 CT scans 3\ct_scans_png\scan_00344.png"]
test_ds   = Test_UnetDataset(img_pth)

compare_models = [
    ("deeplabv3plus", "efficientnet-b3"),
    ("deeplabv3plus", "mit_b1"),
    ("segformer",     "efficientnet-b3"),
    ("segformer",     "mit_b1"),
    ("unetpp",        "efficientnet-b3"),
]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------
# UI
# -----------------------------
st.markdown("""
    <style>
        .main { background-color: #f9fafc; }
        h1, h2, h3 { color: #0073e6; }
        .stButton>button {
            background-color: #0073e6;
            color: white;
            border-radius: 8px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            min-width: 500px;
            max-width: 500px;
        }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("🧠 Segmentation Model Playground")
    st.markdown("Upload an image, choose a model, and see the result.")

    # >>> UPLOAD IMG
    inp = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "tif", "dcm"])

    # >>> CHOOSE MODEL
    model_choice = st.selectbox(
        "Available [Decoder(Model-type) | Encoder(Backbone)]",
        [f'{d}|{e}' for d, e in compare_models]
    )

    run = st.button("Run Inference")    # To start predicting

# -----------------------------
# Logic
# -----------------------------
col1, col2 = st.columns(2)
if inp is not None:
    
    # Transform image before give it to the model
    og_img = Image.open(inp).convert("RGB")
    with col1:
        st.image(og_img, caption="🖼 Original Image", width="stretch")
    inp = test_ds.__getitem__(img_in=og_img)['image'].unsqueeze(0).to(device)
    
    # Run and show predictions
    if run:
        decoder, encoder = model_choice.split("|")
        with st.spinner("Running prediction..."):
            pred_msk = get_prediction(model_dir, decoder, encoder, exp_name, inp)
            pred_msk = np.array(pred_msk)
            
            pred_msk_img = Image.fromarray((pred_msk * 255).astype(np.uint8)).resize(og_img.size)
            pred_msk_np = np.array(pred_msk_img)
            
            overlay = np.array(og_img).copy()
            overlay[pred_msk_np > 127] = [255, 0, 0]
            
            with col2:
                st.image(overlay, caption="🎯 Segmentation Overlay", width="stretch")
    else:
        st.info("👉 Select model and click **Run Inference** to generate overlay.")