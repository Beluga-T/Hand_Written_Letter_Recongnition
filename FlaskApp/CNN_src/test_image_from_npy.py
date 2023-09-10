import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the model
model = load_model('my_model.keras')

# Load test images and labels
test_images = np.load('../preprocess_images_data_npy/test_images.npy')
test_labels = np.load('../preprocess_images_data_npy/test_labels.npy')

# Adjust the labels to be 0-indexed
test_labels = test_labels-1

# Predict a single image
index = 52  # Change this to predict a different image
image = test_images[index].reshape(1, 28, 28, 1)
predicted_probs = model.predict(image)
predicted_label = np.argmax(predicted_probs)

print(f"Predicted Label: {predicted_label}")
print(f"True Label: {test_labels[index]}")
