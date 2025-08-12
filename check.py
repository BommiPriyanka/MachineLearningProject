import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
import sys

MODEL_PATH = 'glaucoma_model_mobilenetv2.h5'
IMG_PATH = 'Yes2.jpg'

# Load model
model = load_model(MODEL_PATH)

# Check image file
if not os.path.exists(IMG_PATH):
    print(f"Error: Image not found -> {IMG_PATH}")
    sys.exit(1)

# Load and preprocess image
img = image.load_img(IMG_PATH, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)[0][0]

# Output result
if prediction >= 0.5:
    print(f"🟢 Normal Eye (Confidence: {1 - prediction:.2f})")
else:
    print(f"🔴 Glaucoma Detected (Confidence: {prediction:.2f})")
    