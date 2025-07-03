import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import numpy as np
import cv2

# Load the model
model = MobileNetV2(weights='imagenet')

# Load and preprocess the image
def load_and_prepare_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize for MobileNetV2
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(np.expand_dims(img, axis=0))
    return img

# Predict the food
def predict_food(image_path):
    image = load_and_prepare_image(image_path)
    prediction = model.predict(image)
    decoded = decode_predictions(prediction, top=1)[0][0]
    label = decoded[1]
    confidence = decoded[2] * 100
    print(f"Prediction: {label} ({confidence:.2f}%)")

# Example usage
predict_food("data/test-images/pizza.jpg")  # Replace with your image path
