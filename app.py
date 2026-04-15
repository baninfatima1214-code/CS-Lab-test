import streamlit as st
import os
from PIL import Image
import numpy as np

# Page config
st.set_page_config(page_title="AI Detector", layout="wide")

st.title("🚀 AI Object Detector (Offline Model)")

# 1. Force install/import logic - Move this inside a function or handle carefully
try:
    from ultralytics import YOLO
    IMPORT_SUCCESS = True
except ImportError:
    IMPORT_SUCCESS = False
    st.error("❌ Ultralytics library missing! Please ensure 'ultralytics' is in requirements.txt and Reboot the app.")

if IMPORT_SUCCESS:
    # 2. Cache the model so it doesn't reload every time
    @st.cache_resource
    def load_my_model():
        # yolov8n.pt will auto-download on first run
        return YOLO('yolov8n.pt') 

    try:
        model = load_my_model()
        st.success("✅ AI Model Loaded Successfully!")
    except Exception as e:
        st.error(f"Model loading error: {e}")
        st.stop() # Stop execution if model fails

    # 3. Simple UI
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB") # Convert to RGB to avoid alpha channel issues
        st.image(image, caption="Uploaded Image", width=400)
        
        if st.button("Analyze Now"):
            with st.spinner("AI is thinking..."):
                # Convert PIL to numpy format
                img_array = np.array(image)
                
                # Run Detection
                results = model(img_array)
                
                # Show results
                res_plotted = results[0].plot()
                st.image(res_plotted, caption="AI Result", use_container_width=True)
                
                # List found objects
                st.write("### Objects Found:")
                if len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        label = model.names[int(box.cls[0])]
                        st.write(f"- {label}")
                else:
                    st.write("No objects detected.")
else:
    st.info("Waiting for libraries to be installed...")
