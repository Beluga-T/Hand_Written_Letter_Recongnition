import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load the models
model_ann = tf.keras.models.load_model('trained_models/my_model_best_ANN.keras')
model_cnn = tf.keras.models.load_model('trained_models/my_model_CNN.keras')

# Add a softmax layer to the ANN model for prediction
model_ann_softmax = tf.keras.Sequential([model_ann, tf.keras.layers.Softmax()])

# Load the test data
test_images = np.load('preprocess_images_data_npy/test_images.npy')
test_labels = np.load('preprocess_images_data_npy/test_labels.npy')
# Class names for EMNIST letters are from 'a' to 'z' (1-based index)
class_names = [chr(c) for c in range(ord('a'), ord('z') + 1)]
test_labels = test_labels - 1
# Make predictions using the ANN model with softmax
ann_predictions = model_ann_softmax.predict(test_images)
ann_predicted_labels = np.argmax(ann_predictions, axis=1)

# Make predictions using the CNN model
cnn_predictions = model_cnn.predict(test_images)
cnn_predicted_labels = np.argmax(cnn_predictions, axis=1)

# Generate confusion matrices
ann_confusion = confusion_matrix(test_labels, ann_predicted_labels)
cnn_confusion = confusion_matrix(test_labels, cnn_predicted_labels)

# Plot and save the ANN confusion matrix using Seaborn
plt.figure(figsize=(12, 10))
sns.heatmap(ann_confusion, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title('ANN Confusion Matrix')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.savefig('./static/ANN_confusion_matrix.png')

# Plot and save the CNN confusion matrix using Seaborn
plt.figure(figsize=(12, 10))
sns.heatmap(cnn_confusion, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title('CNN Confusion Matrix')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.savefig('./static/CNN_confusion_matrix.png')