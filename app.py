import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Page Configuration
st.set_page_config(page_title="AI Vision Detector", layout="centered")

# Custom Styling for User Friendly UI
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Application Title
st.title("🔍 Smart Object Detector")
st.write("Apni image upload karein aur AI usay analyze kar k objects bataye ga.")

# Load Pre-trained Model (No API needed)
# Ye line automatically model download kar legi jab pehli bar chale gi
@st.cache_resource
def load_model():
    return YOLO('yolo11n.pt')  # 'n' stands for nano (fast and lightweight)

model = load_model()

# File Uploader
uploaded_file = st.file_uploader("Image select karein...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Uploaded Image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("Original Image")
        st.image(image, use_container_width=True)

    # Perform Detection
    if st.button('Analyze Image'):
        with st.spinner('AI analysis kar raha hai...'):
            # Convert image to numpy array for YOLO
            img_array = np.array(image)
            
            # Predict
            results = model(img_array)
            
            # Plot results (bounding boxes etc.)
            res_plotted = results[0].plot()
            
            with col2:
                st.success("AI Result")
                st.image(res_plotted, caption='Detected Objects', use_container_width=True)
                
            # Show summary of detected objects
            st.write("### Detection Summary:")
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    label = model.names[class_id]
                    conf = round(float(box.conf[0]) * 100, 1)
                    st.write(f"- ✅ **{label.capitalize()}** (Confidence: {conf}%)")
            else:
                st.write("Koi object detect nahi hua.")

else:
    st.warning("Baraye meharbani koi image upload karein.")

st.sidebar.title("About")
st.sidebar.info("Ye app Computer Vision (YOLO) use karti hai. Is mein koi external API use nahi ho rahi, sab kuch model file ke zariye ho raha hai.")
