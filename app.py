import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("flower_model.keras")

# Class names
class_names = ["daisy", "rose", "sunflower"]

st.title("🌸 Flower Classification Using CNN")

uploaded_file = st.file_uploader(
    "Upload a flower image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file)

    st.image(
        img,
        caption="Uploaded Image",
        width=300
    )

    # Resize according to training size
    img = img.resize((64, 64))

    # Convert image to array
    img_array = image.img_to_array(img)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.success(
        f"Prediction: {class_names[predicted_class]}"
    )

    st.info(
        f"Confidence: {confidence:.2f}%"
    )