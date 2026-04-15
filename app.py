import streamlit as st
import os

# 1. Force install/import logic
try:
    from ultralytics import YOLO
except ImportError:
    st.error("Ultralytics library missing. Please check requirements.txt")

from PIL import Image
import numpy as np

# Page config
st.set_page_config(page_title="AI Detector", layout="wide")

st.title("🚀 AI Object Detector (Offline Model)")

# 2. Cache the model so it doesn't reload every time
@st.cache_resource
def load_my_model():
    # This will download the tiny 6MB model to the server
    return YOLO('yolov8n.pt') 

try:
    model = load_my_model()
    st.success("AI Model Loaded Successfully!")
except Exception as e:
    st.error(f"Model loading error: {e}")

# 3. Simple UI
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=400)
    
    if st.button("Analyze Now"):
        with st.spinner("Processing..."):
            # Convert PIL to OpenCV format
            img_array = np.array(image)
            
            # Run Detection
            results = model(img_array)
            
            # Show results
            res_plotted = results[0].plot()
            st.image(res_plotted, caption="AI Result", use_container_width=True)
            
            # List found objects
            st.write("### Objects Found:")
            for box in results[0].boxes:
                label = model.names[int(box.cls[0])]
                st.write(f"- {label}")
