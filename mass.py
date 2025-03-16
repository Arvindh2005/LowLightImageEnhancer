import streamlit as st
import cv2 as cv
import numpy as np
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from PIL import Image

model = tf.keras.models.load_model("real_mass.h5")

pixel = 256
def ExtractTestInput(image):
    image = np.array(image)  
    img = cv.cvtColor(image, cv.COLOR_RGB2BGR)  
    img = cv.resize(img, (pixel, pixel))
    return img.reshape(1, pixel, pixel, 3)


def display_images(test_image_path):
    
    img_ = ExtractTestInput(test_image_path)
    Prediction = model.predict(img_)
    img_ = img_.reshape(pixel, pixel, 3)

    Prediction = Prediction.reshape(pixel, pixel, 3)
    img_[:, :, :] = Prediction[:, :, :]
    return img_


st.title("Low-Light Image Enhancement Web App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    
    if st.button("Enhance Image"):
        enhanced_img = display_images(image)
        st.image(enhanced_img, caption="Enhanced Image", use_container_width=True)





