import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# Page layout
st.markdown("""
   <div style='text-align: center; margin-bottom: 1.5em;'>
      <h1 style='font-size:2.5em; margin-bottom:0.2em; word-break:break-word; white-space:normal; line-height:1.1;'>
         🍽️ <span style='color:#FF9800;'>Food</span> <span style='color:#43A047;'>Recognition</span> <span style='color:#1976D2;'>System</span>
      </h1>
      <p style='font-size:1.15em; font-weight:500; color:#888888; margin-top:0.5em;'>
         Upload a food photo and let our AI guess what's on your plate!<br>
         <span style='color:#FF9800;'>Fast</span> • <span style='color:#43A047;'>Accurate</span> • <span style='color:#1976D2;'>Fun</span>
      </p>
   </div>
   <style>
      @media (max-width: 600px) {
         h1 {
            font-size: 1.5em !important;
            line-height: 1.15 !important;
            word-break: break-word !important;
            white-space: normal !important;
         }
      }
   </style>
""", unsafe_allow_html=True)

with st.expander("ℹ️ How it works", expanded=False):
    st.markdown("""
    - Upload a clear image of a food item.
    - The system first tries to recognize it using a custom-trained model.
    - If uncertain, it falls back to a general AI model for a best guess.
    - Supported formats: **jpg, jpeg, png**
    """)

# --- Load Models ---
try:
    custom_model = tf.keras.models.load_model("model/model.keras")
except Exception as e:
    st.error("❌ Failed to load custom model.")
    st.stop()

imagenet_model = MobileNetV2(weights="imagenet")
class_names = ['Amala','cheese','Eggroll', 'Egusi soup', 'Hotdog', 'Jollof rice', 'Pizza', 'Puff puff','Rice']  # Match your training labels

# --- Upload Image ---
uploaded_file = st.file_uploader("📤 Choose a food image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Decode image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Uploaded Image", use_container_width=True)

    # Resize & prepare for custom model
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_input = np.expand_dims(img_resized.astype("float32") / 255.0, axis=0)

    # Predict with custom model
    custom_preds = custom_model.predict(img_input)[0]
    max_idx = np.argmax(custom_preds)
    conf_score = custom_preds[max_idx]
    label = class_names[max_idx]

    if conf_score >= 0.6:
        st.success(f"🧠 Custom Model Prediction: **{label}** ({conf_score * 100:.2f}%)")
    else:
        st.warning(f"⚠️ Low confidence from custom model ({conf_score * 100:.2f}%). Trying fallback model...")
        fallback_img = preprocess_input(np.expand_dims(img_resized, axis=0))
        fallback_preds = imagenet_model.predict(fallback_img)
        fallback_label, fallback_conf = decode_predictions(fallback_preds, top=1)[0][0][1:]
        st.info(f"🔍 Alternative Prediction: **{fallback_label}** ({fallback_conf * 100:.2f}%)")
