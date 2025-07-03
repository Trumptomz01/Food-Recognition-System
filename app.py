import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

st.set_page_config(
    page_title="Food Recognition System",  # Browser tab title
    page_icon="🤖",                           # Tab icon (emoji or link)
    layout="centered",                         # Or "wide"
    initial_sidebar_state="auto"               # Or "collapsed" / "expanded"
)
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- Load both models ---
custom_model = tf.keras.models.load_model("model/")
imagenet_model = MobileNetV2(weights="imagenet")              # default MobileNetV2

# --- Your 5 custom food classes ---
CONFIDENCE_THRESHOLD = 0.90  
class_names = ['Amala', 'Cheese', 'Eggroll', 'Egusi Soup', 'Hotdog', 'Jollof Rice', 'Pizza', 'Puff puff', 'Rice']

# --- UI ---
# --- Improved UI Header ---
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

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
   # Read and show image
   file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
   img_bgr = cv2.imdecode(file_bytes, 1)
   img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
   st.image(img_rgb, caption="Uploaded Image", use_container_width=True)

    # --- Resize and preprocess for custom model ---
   img_resized = cv2.resize(img_rgb, (224, 224))
   img_custom = img_resized.astype("float32") / 255.0
   img_custom = np.expand_dims(img_custom, axis=0)

   # Predict with your model
   preds_custom = custom_model.predict(img_custom)
   idx_custom = np.argmax(preds_custom[0])
   conf_custom = preds_custom[0][idx_custom]
   label_custom = class_names[idx_custom]

   if conf_custom >= CONFIDENCE_THRESHOLD:
      st.success(f"🧠 Custom Model Prediction: **{label_custom}** ({conf_custom * 100:.2f}%)")
   else:
      # fallback to ImageNet prediction
      img_mobilenet = preprocess_input(np.expand_dims(img_resized, axis=0))
      preds_mobilenet = imagenet_model.predict(img_mobilenet)
      label_imagenet, confidence_imagenet = decode_predictions(preds_mobilenet, top=1)[0][0][1:]

      st.warning("❗ Low confidence from custom model. Using fallback:")
      st.info(f"🔍 Alternative Prediction: **{label_imagenet}** ({confidence_imagenet * 100:.2f}%)")
      st.info(f"🧠 Custom Model Guess: **{label_custom}** ({conf_custom * 100:.2f}%)")
