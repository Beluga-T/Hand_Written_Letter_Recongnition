
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
# list_datasets()
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization,SeparableConv2D
from keras.models import Sequential
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedShuffleSplit
import os
# set the gpu to use
# print gpu available list
print("GPU list:", tf.config.experimental.list_physical_devices('GPU'))
# if no gpu available, use cpu
if len(tf.config.experimental.list_physical_devices('GPU')) == 0:
    print("No GPU available, using CPU instead.")
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
else:
    print("GPU available.")
    #choose gpu to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'

    plt.xlabel("{:2.0f}% sure that is a {} \n (True value is {})".format(100 * np.max(predictions_array),
                                                                         class_names[predicted_label],
                                                                         class_names[true_label]),
                                                                         color=color, fontsize=10)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(26), class_names, fontsize=10)
    plt.yticks([])
    thisplot = plt.bar(range(26), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('green')


def extract_and_preprocess(images_and_labels):
    images = []
    labels = []

    with tf.device('/CPU:0'):
        for image, label in tqdm(tfds.as_numpy(images_and_labels), desc="Preprocessing"):
            # Normalize the image to [0, 1]
            image = image.astype('float32') / 255.0
            # EMNIST images are rotated by 90 degrees, fix the rotation
            image = tf.image.rot90(image).numpy()
            # image need to flipped upside down
            image = np.flipud(image)

            images.append(image)
            labels.append(label)
        return (np.array(images), np.array(labels))

# # prepare the test_images for single prediction
# image_for_test = plt.imread('./test_images_png/5.png')
# image_for_test = image_for_test.astype('float32') / 255.0
# image_for_test = image_for_test.reshape(-1, 28, 28, 1)


# Load the EMNIST letters dataset
(train_data, test_data), ds_info = tfds.load(
    'emnist/letters',
    split=['train', 'test'],
    as_supervised=True,  # Returns tuple (img, label) instead of dict
    with_info=True      # Provides metadata about the dataset
)
print("EMNIST letters dataset loaded.")


# Extract the images and labels and apply preprocessing
all_images, all_labels = extract_and_preprocess(train_data)

# Flatten the images for stratified split
all_images_flat = all_images.reshape(-1, 28*28)

# Use StratifiedShuffleSplit to ensure all classes are in both train and test sets
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)  # 20% test size, adjust as needed
for train_index, test_index in sss.split(all_images_flat, all_labels):
    train_images, test_images = all_images[train_index], all_images[test_index]
    train_labels, test_labels = all_labels[train_index], all_labels[test_index]

# Reshape the images from (28, 28) to (28, 28, 1)
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

# Save train_images, train_labels, test_images, test_labels as npy files
np.save('../preprocess_images_data_npy/train_images.npy', train_images)
np.save('../preprocess_images_data_npy/train_labels.npy', train_labels)
np.save('../preprocess_images_data_npy/test_images.npy', test_images)
np.save('../preprocess_images_data_npy/test_labels.npy', test_labels)

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

# save 10 test_images as png files with exaclty 28*28 pixels
#
for i in range(10):
    # normalize the image
    test_images[i] = test_images[i].astype('float32') / 255.0
    # save under ./test_images_png folder
    plt.imsave('../test_images_png/' + str(i) + '.png', test_images[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.imshow(train_images[i].reshape(28, 28), cmap=plt.cm.binary)


# Normalize the images


# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(train_images)


# Model Architecture
model = Sequential()
model.add(BatchNormalization(input_shape=(28, 28, 1)))
model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='gelu', activity_regularizer = regularizers.L1L2(l1=1e-6, l2=1e-6),input_shape=(28, 28, 1)))
model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='gelu', activity_regularizer = regularizers.L1L2(l1=1e-6, l2=1e-6)))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(filters=64,kernel_size=3, padding='same', activation='gelu',activity_regularizer = regularizers.L1L2(l1=1e-6, l2=1e-6)))
model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='gelu', activity_regularizer = regularizers.L1L2(l1=1e-6, l2=1e-6)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu', activity_regularizer = regularizers.L1L2(l1=1e-6, l2=1e-6)))
model.add(Dense(128, activation='relu', activity_regularizer = regularizers.L1L2(l1=1e-6, l2=1e-6)))
model.add(Dense(26, activation='softmax'))

# Compile the model
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

#plot the model
keras.utils.plot_model(model, "cnn_model.png", show_shapes=True)

# Callbacks
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=6, verbose=1, factor=0.5, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=20, verbose=1, restore_best_weights=True)

# Train the model
history = model.fit(datagen.flow(train_images, train_labels, batch_size=2048),
                    epochs=300,
                    validation_data=(test_images, test_labels),
                    callbacks=[learning_rate_reduction, early_stopping])
# Save the model
model.save('../trained_models/my_model_CNN.keras')


# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
print('Test loss:', test_loss)

# Make predictions
predictions = model.predict(test_images)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Evolution')


plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy Evolution')

#save the plot
plt.savefig('Accuracy_and_Loss.png', dpi=335)  # Save the figure



# for i in range(10):  # print the first 10 predictions
#     print("Predicted probabilities:", predictions[i])
#     print("Predicted class:", np.argmax(predictions[i]))
#     print("True class:", test_labels[i])
#     print("------")
# print(np.sum(predictions[0]))  # Should be close to 1


# Print the image
num_rows = 5
num_cols = 5
num_images = num_rows * num_cols

# Increase the figure size
plt.figure(figsize=(15, 15))

# random choose 9 images
indices = np.random.choice(len(test_images), num_images, replace=False)
for i, index in enumerate(indices):
    # Plot the image
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(index, predictions[index], test_labels, test_images)

    # Plot the prediction bar chart
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(index, predictions[index], test_labels)

# Adjust spacing
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
plt.tight_layout()
plt.savefig('output_image.png', dpi=335)  # Save the figure
plt.show()

#make a prediction on a single image
# choose an image from the test set and save it as test_image.png 28*28



