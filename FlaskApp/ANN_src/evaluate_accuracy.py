import numpy as np
import tensorflow as tf


def load_data():
    # Load data
    train_images = np.load('../preprocess_images_data_npy/train_images.npy')
    train_labels = np.load('../preprocess_images_data_npy/train_labels.npy')
    test_images = np.load('../preprocess_images_data_npy/test_images.npy')
    test_labels = np.load('../preprocess_images_data_npy/test_labels.npy')

    # Adjust labels
    train_labels = train_labels - 1
    test_labels = test_labels - 1

    return train_images, train_labels, test_images, test_labels


def evaluate_model(model_path, test_images, test_labels, wrap_softmax=False):
    # Load model
    model_ann = tf.keras.models.load_model(model_path)

    if wrap_softmax:
        model = tf.keras.Sequential([model_ann, tf.keras.layers.Softmax()])
    else:
        model = model_ann

    # Make predictions
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)

    correct_predictions = np.sum(predicted_labels == test_labels)
    total_predictions = test_labels.shape[0]

    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy


def main():
    # Load the data
    _, _, test_images, test_labels = load_data()

    # Ask user for model path
    model_path = "../trained_models/my_model_best_ANN.keras"

    accuracy = evaluate_model(model_path, test_images, test_labels, wrap_softmax=True)
    print(f"ANN Test Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
