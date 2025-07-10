import os
import gdown

import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from model import SimpleModel 
# from dataset import ArtportalenDataModule  # Import your dataset
from pytorch_lightning import Trainer
from inference import run_inference_and_gradcam
from ultralytics import YOLO
from streamlit_cropper import st_cropper
import yaml
from feedback_logger import log_feedback

from model import SimpleModel

REQUIRE_SINGLE_EAGLE = False      # False ➜ accept first mask regardless
FEEDBACK_ENABLED     = False   # set to False to hide / skip feedback

# Load config from YAML file
with open('config-local-artportalen.yml', 'r') as file:
    config = yaml.safe_load(file)

# Set up Streamlit page configuration
st.set_page_config(page_title="Eagle Age Prediction", page_icon=":eagle:", initial_sidebar_state="auto")

# Hide Streamlit's style elements
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Define label mapping
label_mapping = {0: '1K', 1: '2K', 2: '3K', 3: '4K', 4: '5K_plus'}


@st.cache_resource
def load_yolo_model():
    yolo_weights_path = "yolo11x-seg.pt"
    model = YOLO(yolo_weights_path)
    return model

@st.cache_resource
def load_classifier_model():
    model_path = "model.ckpt"
    file_id = "1V6QpPNxaYAymii7Sc2OY1QX47pU80-iN"  #
    if not os.path.exists(model_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)
    checkpoint_data = torch.load(model_path, map_location=torch.device('cpu'))
    return checkpoint_data



# Function to unnormalize the image
def unnormalize(x, mean, std):
    unnormalized_x = x.clone().detach()
    for t, m, s in zip(unnormalized_x, mean, std):
        t.mul_(s).add_(m)
    return unnormalized_x

# Function to visualize GradCAM
def visualize_gradcam(model, image, label, mean, std, target_layer):
    image.requires_grad_()
    model.eval()
    output = model(image.unsqueeze(0))
    prediction = torch.argmax(output, dim=1).item()
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(label.item())]
    grayscale_cam = cam(input_tensor=image.unsqueeze(0), targets=targets)[0]
    unnormalized_img = unnormalize(image.cpu().detach(), mean, std).permute(1, 2, 0).numpy()
    unnormalized_img = np.clip(unnormalized_img, 0, 1)
    visualization = show_cam_on_image(unnormalized_img, grayscale_cam, use_rgb=True)
    return unnormalized_img, visualization, prediction

def full_reset():
    # Clear everything and force uploader reset
    st.session_state.clear()
    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1
    st.rerun()



# Sidebar content
with st.sidebar:
    st.title("Golden Eagle AI Age Prediction Tool")
    st.subheader("Upload any image of a golden eagle to predict its age. The model is trained on images of golden eagles from the Swedish Artportalen database.")

# Unique key to control uploader state
uploader_key = st.session_state.get("uploader_key", 0)
file = st.file_uploader("Upload an image", type=["jpg", "png"], label_visibility="collapsed", key=uploader_key)


# Define normalization parameters for image preprocessing
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

# Define and load the model
model = SimpleModel(config=config, pretrained=False, num_classes=5)
try:
    checkpoint_data = load_classifier_model()
    model.load_state_dict(checkpoint_data["state_dict"])
except Exception as e:
    st.error(f"Could not load checkpoint: {e}")
    st.warning("Using untrained model - results will be random!")

yolo_model = load_yolo_model()


if file is None:
    st.text("Please upload an image file")
else:
    # Fix file handling issue
    try:
        image = Image.open(file).convert("RGB")
        st.image(image, caption="Original Image", use_container_width=True)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        st.stop()

    # Run YOLO segmentation with loading spinner
    with st.spinner("Running YOLO segmentation…"):
        results = yolo_model(image)

    # Collect masks & object count
    masks = results[0].masks.data if results[0].masks is not None else None
    n_objects = len(results[0].boxes) if results[0].boxes is not None else 0

    # Decide which mask to use
    mask = None
    if masks is not None:
        if REQUIRE_SINGLE_EAGLE:
            # Only proceed if YOLO found exactly one eagle
            if n_objects == 1:
                mask = (masks[0].cpu().numpy() > 0.5).astype(np.uint8)
        else:
            # Always take the first mask, even if YOLO found several
            mask = (masks[0].cpu().numpy() > 0.5).astype(np.uint8)
            if n_objects > 1:
                st.info(f"YOLO detected {n_objects} eagles – using the first one automatically.")

    # ── ──Segmentation Success path ──────────────────────────────────────────────
    if mask is not None:
        # Resize mask to match image size
        with st.spinner("Processing mask..."):
            mask_resized = Image.fromarray(mask * 255).resize(image.size, resample=Image.NEAREST)
            mask_resized = np.array(mask_resized) // 255  # back to 0/1
        # Show masked image
        image_np = np.array(image)
        masked_image_np = image_np * mask_resized[..., None]
        st.image(masked_image_np, caption="Masked Image", use_container_width=True)
        
        # Run Classification button
        if st.button("Run Classification"):
            input_image = Image.fromarray(masked_image_np.astype(np.uint8))
            with st.spinner("Running classification and GradCAM..."):
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])
                image_tensor = transform(input_image).unsqueeze(0)
                image_tensor = image_tensor.squeeze(0)
                pred, gradcam_img = run_inference_and_gradcam(model, image_tensor, mean, std)
            st.session_state["gradcam_img"] = gradcam_img
            st.session_state["pred"] = pred
            st.session_state["classification_done"] = True
            st.session_state["feedback_given"] = False
        # Show results if classification was run
        if st.session_state.get("classification_done", False):
            st.image(st.session_state["gradcam_img"], caption=f"GradCAM Visualization - Predicted Age: {label_mapping[st.session_state['pred']]}", use_container_width=True)
            st.write(f"**Predicted Age Group: {label_mapping[st.session_state['pred']]}**")
            if FEEDBACK_ENABLED:
                # Feedback section
                st.write("---")
                st.write("**Was this prediction correct?**")
                if not st.session_state.get("feedback_given", False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("✅ Correct", key="correct_1"):
                            st.session_state["pending_feedback"] = ("correct", file.name if file else None)

                    with col2:
                        if st.button("❌ Incorrect", key="incorrect_1"):
                            st.session_state["pending_feedback"] = ("incorrect", file.name if file else None)
                    with col3:
                        if st.button("❓ Unsure", key="unsure_1"):
                            st.session_state["pending_feedback"] = ("unsure", file.name if file else None)
                else:
                    st.success("Thank you for your feedback!")
            # Start Over button
            st.write("---")
            if st.button("🔄 Start Over", key="start_over_1"):
                full_reset()

    
    # ──Segmentation Failure path ──────────────────────────────────────────────
    else:
        if REQUIRE_SINGLE_EAGLE:
            # Your existing cropper UI goes here
            st.warning("YOLO found no single-eagle mask – please crop.")
            # (paste the whole cropper code block)
        else:
            st.error("YOLO produced no masks; cannot continue.")

        
        # Initialize session state variables
        if 'show_cropper' not in st.session_state:
            st.session_state.show_cropper = False
        if 'cropped_img' not in st.session_state:
            st.session_state.cropped_img = None
        if 'masked_cropped_np' not in st.session_state:
            st.session_state.masked_cropped_np = None
        if 'cropped_mask_ready' not in st.session_state:
            st.session_state.cropped_mask_ready = False
        if 'classification_ready' not in st.session_state:
            st.session_state.classification_ready = False
        
        # Show Crop Image button if cropper is not active
        if not st.session_state.show_cropper:
            if st.button("Crop Image"):
                st.session_state.show_cropper = True
                st.session_state.cropped_mask_ready = False
                st.session_state.classification_ready = False
                st.rerun()
        
        # Show cropping interface if activated
        if st.session_state.show_cropper:
            from streamlit_cropper import st_cropper
            cropped_img = st_cropper(
                image,
                aspect_ratio=None,
                box_color='#0000FF',
                return_type='image',
                realtime_update=True,
                key='cropper'
            )
            if cropped_img is not None:
                st.image(cropped_img, caption="Cropped Image", use_container_width=True)
                st.session_state.cropped_img = cropped_img
                if st.button("Rerun Segmentation"):
                    with st.spinner("Running YOLO segmentation on cropped image..."):
                        results_cropped = yolo_model(st.session_state.cropped_img)
                        n_objects_cropped = len(results_cropped[0].boxes) if results_cropped[0].boxes is not None else 0
                        mask_cropped = None
                        if results_cropped[0].masks is not None and n_objects_cropped == 1:
                            mask_cropped = results_cropped[0].masks.data[0].cpu().numpy()
                            mask_cropped = (mask_cropped > 0.5).astype(np.uint8)
                    
                    if mask_cropped is not None:
                        mask_resized_cropped = Image.fromarray(mask_cropped * 255).resize(st.session_state.cropped_img.size, resample=Image.NEAREST)
                        mask_resized_cropped = np.array(mask_resized_cropped) // 255
                        cropped_np = np.array(st.session_state.cropped_img)
                        masked_cropped_np = cropped_np * mask_resized_cropped[..., None]
                        st.image(masked_cropped_np, caption="Masked Cropped Image", use_container_width=True)
                        st.session_state.masked_cropped_np = masked_cropped_np
                        st.session_state.cropped_mask_ready = True
                        st.session_state.classification_ready = False
                    else:
                        st.error(f"YOLO still detected {n_objects_cropped} objects on cropped image. Please try cropping more precisely.")

        # Only show Run Classification if mask is ready
        if st.session_state.cropped_mask_ready:
            if st.button("Run Classification"):
                input_image = Image.fromarray(st.session_state.masked_cropped_np.astype(np.uint8))
                with st.spinner("Running classification and GradCAM..."):
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std)
                    ])
                    image_tensor = transform(input_image).unsqueeze(0)
                    image_tensor = image_tensor.squeeze(0)
                    pred, gradcam_img = run_inference_and_gradcam(model, image_tensor, mean, std)
                st.session_state["gradcam_img"] = gradcam_img
                st.session_state["pred"] = pred
                st.session_state["classification_done"] = True
                st.session_state["feedback_given"] = False
            # Show results if classification was run
            if st.session_state.get("classification_done", False):
                st.image(st.session_state["gradcam_img"], caption=f"GradCAM Visualization - Predicted Age: {label_mapping[st.session_state['pred']]}", use_container_width=True)
                st.write(f"**Predicted Age Group: {label_mapping[st.session_state['pred']]}**")
                # Feedback section
                st.write("---")
                st.write("**Was this prediction correct?**")
                if not st.session_state.get("feedback_given", False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("✅ Correct", key="correct_2"):
                            log_feedback(label_mapping[st.session_state['pred']], "correct", file.name if file else None)
                            st.session_state["feedback_given"] = True
                    with col2:
                        if st.button("❌ Incorrect", key="incorrect_2"):
                            log_feedback(label_mapping[st.session_state['pred']], "incorrect", file.name if file else None)
                            st.session_state["feedback_given"] = True
                    with col3:
                        if st.button("❓ Unsure", key="unsure_2"):
                            log_feedback(label_mapping[st.session_state['pred']], "unsure", file.name if file else None)
                            st.session_state["feedback_given"] = True
                else:
                    st.success("Thank you for your feedback!")
                # Start Over button
                st.write("---")
                if st.button("🔄 Start Over", key="start_over_2"):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()


# Handle deferred feedback logging (should run after all UI)
if FEEDBACK_ENABLED and "pending_feedback" in st.session_state:
    feedback_type, image_name = st.session_state["pending_feedback"]
    log_feedback(label_mapping[st.session_state['pred']], feedback_type, image_name)
    st.session_state["feedback_given"] = True
    del st.session_state["pending_feedback"]
    st.rerun()
