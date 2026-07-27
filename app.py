import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("flower_model.keras")

# Class names (change these if your dataset has different folders)
class_names = ["daisy", "rose", "sunflower"]

st.title("🌸 Flower Classification Using CNN")

uploaded_file = st.file_uploader(
    "Upload a flower image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {class_names[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}%")