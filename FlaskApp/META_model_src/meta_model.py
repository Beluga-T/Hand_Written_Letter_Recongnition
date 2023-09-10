import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
cnn_model = tf.keras.models.load_model('../trained_models/my_model_CNN.keras')
model_ann = tf.keras.models.load_model('../trained_models/my_model_best_ANN.keras')
model_ann_softmax = tf.keras.Sequential([model_ann, tf.keras.layers.Softmax()])  # Wrap the ANN model once here

# load train_images, train_labels, test_images, test_labels as npy files
train_images = np.load('../preprocess_images_data_npy/train_images.npy')
train_labels = np.load('../preprocess_images_data_npy/train_labels.npy')
test_images = np.load('../preprocess_images_data_npy/test_images.npy')
test_labels = np.load('../preprocess_images_data_npy/test_labels.npy')
# Class names for EMNIST letters are from 'a' to 'z' (1-based index)
class_names = [chr(c) for c in range(ord('a'), ord('z') + 1)]
# Adjust the labels to be 0-indexed
train_labels = train_labels - 1
test_labels = test_labels - 1

# Assuming test_labels is already loaded
unique_classes, counts = np.unique(test_labels, return_counts=True)
# Print the counts for each class
for u_class, count in zip(unique_classes, counts):
    print(f"Class {class_names[u_class]}: {count} samples")

# Generate predictions for training data
cnn_predictions = cnn_model.predict(train_images)
ann_predictions = model_ann_softmax.predict(train_images)


# Combine the predictions for training data
stacked_predictions = np.column_stack((cnn_predictions, ann_predictions))

# Generate predictions for test data
cnn_test_predictions = cnn_model.predict(test_images)
ann_test_predictions = model_ann_softmax.predict(test_images)

# Combine the predictions for test data
stacked_test_predictions = np.column_stack((cnn_test_predictions, ann_test_predictions))

# Train the meta model
meta_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(50, activation='relu', input_dim=stacked_predictions.shape[1]),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

# Compile the meta model
meta_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the meta model
meta_model.fit(stacked_predictions, train_labels, epochs=100, batch_size=4096, validation_data=(stacked_test_predictions, test_labels))


cnn_test_predictions = cnn_model.predict(test_images)
ann_test_predictions = model_ann_softmax.predict(test_images)

stacked_test_predictions = np.column_stack((cnn_test_predictions, ann_test_predictions))

ensemble_predictions = meta_model.predict(stacked_test_predictions)
loss, accuracy = meta_model.evaluate(stacked_test_predictions, test_labels)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")

meta_model.save("../trained_models/MetaModel.keras")

predicted_classes = np.argmax(ensemble_predictions, axis=1)

cm = confusion_matrix(test_labels, predicted_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
#save the plot
plt.savefig('meta_confusion_matrix.png')
plt.show()

print(classification_report(test_labels, predicted_classes, target_names=class_names))