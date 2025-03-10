import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

model = load_model("model-80-3.h5")
IMG_SIZE = (128, 128)
disease_classes = [
    "Eczema", "Psoriasis", "Acne", "Melanoma", 
    "Ringworm", "Rosacea", "Vitiligo"
]
st.set_page_config(page_title="Skin Disease Prediction", page_icon="🩺", layout="centered")

st.title("🩺 Skin Disease Prediction")
st.write("Upload an image of a skin condition, and the model will predict the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    img = Image.open(uploaded_file)
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0 
    img_array = np.expand_dims(img_array, axis=0)  

    with st.spinner(" Analyzing Image..."):
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
    
    st.success(f"Predicted Disease: {disease_classes[predicted_class]}")
    
    confidence = np.max(prediction) * 100
    st.write(f" Confidence: {confidence:.2f}%")

    st.markdown(" About the Disease")
    st.info(f"More information about **{disease_classes[predicted_class]}")