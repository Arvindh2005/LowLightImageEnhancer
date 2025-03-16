# import streamlit as st
# import cv2 as cv
# import numpy as np
# from sklearn.metrics import mean_squared_error
# import tensorflow as tf
# from PIL import Image

# model = tf.keras.models.load_model("real_mass.h5")

# pixel = 256
# def ExtractTestInput(image):
#     image = np.array(image)  
#     img = cv.cvtColor(image, cv.COLOR_RGB2BGR)  
#     img = cv.resize(img, (pixel, pixel))
#     return img.reshape(1, pixel, pixel, 3)


# def display_images(test_image_path):
    
#     img_ = ExtractTestInput(test_image_path)
#     Prediction = model.predict(img_)
#     img_ = img_.reshape(pixel, pixel, 3)

#     Prediction = Prediction.reshape(pixel, pixel, 3)
#     img_[:, :, :] = Prediction[:, :, :]
#     return img_


# st.title("Low-Light Image Enhancement Web App")

# uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_container_width=True)

    
#     if st.button("Enhance Image"):
#         enhanced_img = display_images(image)
#         st.image(enhanced_img, caption="Enhanced Image", use_container_width=True)


import streamlit as st
import cv2 as cv
import numpy as np
import tensorflow as tf
from PIL import Image
import time  

model = tf.keras.models.load_model("real_mass.h5")

PIXEL = 256

def extract_test_input(image):
    image = np.array(image)  
    img = cv.cvtColor(image, cv.COLOR_RGB2BGR)  
    img = cv.resize(img, (PIXEL, PIXEL))
    return img.reshape(1, PIXEL, PIXEL, 3)

def enhance_image(image):
    img_ = extract_test_input(image)
    prediction = model.predict(img_)
    img_ = img_.reshape(PIXEL, PIXEL, 3)
    prediction = prediction.reshape(PIXEL, PIXEL, 3)
    img_[:, :, :] = prediction[:, :, :]
    return img_

st.markdown("""
    <style>
    .title {
        font-size: 72px;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 20px;
    }
    .subtitle {
        font-size: 18px;
        color: #666;
        text-align: center;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="title">✨ Low Light Image Enhancer ✨</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Transform your low-light photos into stunning masterpieces!</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Drop a pic that needs some enhancement 🌟", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    st.image(image, caption="Your Original Snap", use_container_width=True)

    if st.button("Enhance the Magic"):
        with st.spinner("Processing..."):
            time.sleep(1)
            enhanced_img = enhance_image(image)
        
        st.success("Enhancement complete!!")
        st.image(enhanced_img, caption="Enhanced Image", use_container_width=True)

        enhanced_pil = Image.fromarray((enhanced_img * 255).astype(np.uint8))  # Convert back to PIL
        st.download_button(
            label="Download Enhanced Image",
            data=enhanced_pil.tobytes(),
            file_name="enhanced_image.png",
            mime="image/png"
        )

    if st.checkbox("Compare Side-by-Side"):
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Before", use_container_width=True)
        with col2:
            enhanced_img = enhance_image(image)
            st.image(enhanced_img, caption="After", use_container_width=True)

st.markdown("""
    <hr>
""", unsafe_allow_html=True)


