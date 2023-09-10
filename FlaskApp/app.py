import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import traceback
import gc
app = Flask(__name__)
CORS(app)

# Load the model
model = tf.keras.models.load_model('trained_models/my_model_CNN.keras')

# load another model
model_ann = tf.keras.models.load_model('trained_models/my_model_best_ANN.keras')

#load meta model
meta_model = tf.keras.models.load_model('trained_models/MetaModel.keras')


class_names = [chr(c) for c in range(ord('a'), ord('z') + 1)]
@app.route('/')
def index():
    return render_template('index.html')  # Assuming your HTML file is named 'index.html' and is in a 'templates' folder

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from POST request
        data = request.get_json()
        image_data_url = data['image_data_url']
        model_choice = data.get('model_choice', 'cnn')  # Default to 'cnn' if not provided

        # Convert data URL to image
        image_data = base64.b64decode(image_data_url.split(",")[1])
        with open("decoded_image.png", "wb") as f:
            f.write(image_data)

        image = Image.open(io.BytesIO(image_data))

        # Handle transparency: Blend the RGBA image with a white background
        background = Image.new('RGBA', image.size, (255, 255, 255))
        image = Image.alpha_composite(background, image).convert('RGB')

        # Convert to grayscale
        grayscale_np = np.dot(np.array(image)[..., :3], [0.2989, 0.5870, 0.1140])

        # Invert the grayscale values
        grayscale_np = 255.0 - grayscale_np

        # plt.imshow(grayscale_np, cmap='gray')

        # # Save the image without any axis or grid
        # plt.axis('off')
        # plt.grid(False)
        # plt.savefig('grayscale_np.png', bbox_inches='tight', pad_inches=0)

        # plt.show()

        # Normalize to [0, 1]
        image_np = grayscale_np.astype(np.float32) / 255.0
        print("Min:", image_np.min(), "Max:", image_np.max())  # Print min and max values

        image_np = image_np.reshape(-1, 28, 28, 1)  # Reshape to match model input shape

        # Make a prediction based on the chosen model
        if model_choice == 'cnn':
            prediction = model.predict(image_np)
        elif model_choice == 'ann':
            prediction = tf.keras.Sequential([model_ann, tf.keras.layers.Softmax()])
            prediction = prediction.predict(image_np)
        elif model_choice == 'meta':
            cnn_prediction = model.predict(image_np)
            ann_prediction = tf.keras.Sequential([model_ann, tf.keras.layers.Softmax()]).predict(image_np)
            stacked_prediction = np.column_stack((cnn_prediction, ann_prediction))
            prediction = meta_model.predict(stacked_prediction)
        else:
            return jsonify(error='Invalid model choice'), 400

        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        print("model used:", model_choice)
        print("Predicted class:", class_names[predicted_class])
        print("Confidence: {:2.0f}%".format(100 * confidence))

        # Clear memory
        del grayscale_np, image_np
        gc.collect()

        return jsonify({
            'predicted_class': class_names[predicted_class],
            'confidence': confidence,
            'model_used': 'CNN' if model_choice == 'cnn' else ('ANN' if model_choice == 'ann' else 'Meta(CNN+ANN)')

        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify(error=str(e)), 400

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)  # Set debug to False

