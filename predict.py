import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("model/model.keras")  # or use model.keras if you saved with .keras extension

# Set the path to the test folder
test_folder = "data/food-101/test"  # <-- Adjust if yours is different

# Get class names from the training data (must match folder names)
class_names = sorted(os.listdir("data/food-101/train"))  # assuming this exists and is accurate

# Loop through each class folder
for class_dir in os.listdir(test_folder):
    class_path = os.path.join(test_folder, class_dir)
    
    # Skip if it's not a directory
    if not os.path.isdir(class_path):
        continue

    # Loop through images in that class folder
    for img_file in os.listdir(class_path):
        img_path = os.path.join(class_path, img_file)

        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = 100 * np.max(predictions)

        print(f"🧠 Image: {img_file} | Predicted: {predicted_class} ({confidence:.2f}%)")
