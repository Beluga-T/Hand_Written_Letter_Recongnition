import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Load the trained model
model = tf.keras.models.load_model('../trained_models/my_model.keras')
class_names = [chr(c) for c in range(ord('a'), ord('z') + 1)]
# Get the number of units in the last layer
num_classes = model.layers[-1].units

print(f"The model can predict {num_classes} different classes.")

def preprocess_image(image_path):
    # Load the image
    image = Image.open(image_path)

    # # Convert to grayscale
    # image = image.convert('L')
    #
    # # Resize to 28x28
    # image = image.resize((28, 28))

    # Convert to numpy array and normalize
    image_np = np.array(image).astype('float32') / 255.0

    # Reshape to (28, 28, 1)
    image_np = image_np.reshape(28, 28, 1)

    # # Rotate by 90 degrees
    # image_np = tf.image.rot90(image_np).numpy()

    # # Flip upside down
    # image_np = np.flipud(image_np)

    return image_np




def predict_image(image_np):
    # Predict using the model
    prediction = model.predict(np.expand_dims(image_np, axis=0))
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    return class_names[predicted_class], confidence


# Directory containing the PNG files
png_directory = "test_images_png"

# List to store preprocessed images
preprocessed_images = []

# Process each PNG file
for png_file in os.listdir(png_directory):
    if png_file.endswith(".png"):
        image_path = os.path.join(png_directory, png_file)
        preprocessed_image = preprocess_image(image_path)
        preprocessed_images.append(preprocessed_image)

        # Predict the class of the preprocessed image
        predicted_class, confidence = predict_image(preprocessed_image)
        print(f"Image {png_file} is predicted as class {predicted_class} with confidence {confidence:.2f}%")

# Convert list to numpy array
preprocessed_images = np.array(preprocessed_images)

print(preprocessed_images.shape)  # Should print (10, 28, 28, 1) if you have 10 PNG files


