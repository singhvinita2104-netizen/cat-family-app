import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

st.title("Cat Family App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

# Simple descriptions
info = {
    "tiger": "Tiger is a large carnivorous animal known for its stripes.",
    "lion": "Lion is called the king of the jungle and lives in groups.",
    "leopard": "Leopard is a fast and powerful animal with spotted skin.",
    "tabby": "This is a domestic cat commonly found in homes.",
    "tiger_cat": "A type of domestic cat with striped pattern."
}

if uploaded_file is not None:
    # Load and display image
    img = image.load_img(uploaded_file, target_size=(224,224))
    st.image(uploaded_file, caption="Uploaded Image")

    # Load model
    model = MobileNetV2(weights='imagenet')

    # Preprocess image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make prediction
    pred = model.predict(img_array)
    result = decode_predictions(pred, top=1)[0][0][1]

    # Show result
    st.success(f"Prediction: {result}")

    # Show description
    if result in info:
        st.write("About:", info[result])
    else:
        st.write("No detailed info available.")