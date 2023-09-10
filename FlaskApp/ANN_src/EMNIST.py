# Necessary imports
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from keras_tuner import HyperModel

from keras_tuner.tuners import GridSearch
from keras_tuner.tuners import RandomSearch
from sklearn.model_selection import train_test_split

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# Check if GPU is available and set it to be used by TensorFlow
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)



# Functions to plot images and value arrays
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
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
               color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(26), class_names)  # Set x-ticks to be the letters a-z
    plt.yticks([])
    thisplot = plt.bar(range(26), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('green')

# Loading and preprocessing the dataset
(train_data, test_data), ds_info = tfds.load(
    'emnist/letters',
    split=['train', 'test'],
    as_supervised=True,
    with_info=True)

from tqdm import tqdm


from tqdm import tqdm

def extract_and_preprocess(images_and_labels):
    images = []
    labels = []

    # Set the device context to CPU
    with tf.device('/CPU:0'):
        # Wrap the iterable with tqdm to show progress
        for image, label in tqdm(tfds.as_numpy(images_and_labels), desc="preprocessing", unit="items"):
            # Normalize the image to [0, 1]
            image = image.astype('float32') / 255.0
            # EMNIST images are rotated by 90 degrees, fix the rotation
            image = tf.image.rot90(image).numpy()
            # image need to flipped upside down
            image = np.flipud(image)

            images.append(image)
            labels.append(label)

    return (np.array(images), np.array(labels))

class PrintModelSummary(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.model.summary()

class PrintEarlyStopping(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if self.model.stop_training:
            print("\nEarly stopping triggered on epoch {}!".format(epoch + 1))



train_images, train_labels = extract_and_preprocess(train_data)
test_images, test_labels = extract_and_preprocess(test_data)

# Adjust labels and class names
class_names = [chr(c) for c in range(ord('a'), ord('z') + 1)]
train_labels = train_labels - 1
test_labels = test_labels - 1

print("Splitting the dataset...")
# Splitting the dataset
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.3, random_state=42)
val_images, test_images, val_labels, test_labels = train_test_split(val_images, val_labels, test_size=0.5, random_state=42)

class MyHyperModel(HyperModel):

    def build(self, hp):
        model = keras.Sequential()
        model.add(keras.layers.Flatten(input_shape=(28, 28)))

        for i in range(hp.Int('num_layers', 1, 4)):
            model.add(keras.layers.Dense(units=hp.Int('units_' + str(i),
                                                      min_value=32,
                                                      max_value=512,
                                                      step=32),
                                         activation=hp.Choice('activation_' + str(i),
                                                              values=['relu', 'tanh', 'sigmoid']),
                                         kernel_regularizer=keras.regularizers.l2(
                                             hp.Float('l2_reg', min_value=1e-5, max_value=1e-2, sampling='LOG')),
                                         kernel_initializer=hp.Choice('initializer_' + str(i),
                                                                      values=['glorot_uniform', 'he_normal',
                                                                              'lecun_normal'])))

            # Optional Batch Normalization
            if hp.Choice('batch_norm_' + str(i), values=[0, 1]):
                model.add(keras.layers.BatchNormalization())

            model.add(keras.layers.Dropout(rate=hp.Float('dropout_' + str(i),
                                                         min_value=0.0,
                                                         max_value=0.5,
                                                         step=0.1)))

        model.add(keras.layers.Dense(26))

        optimizer_choice = hp.Choice('optimizer', ['adam', 'adamax', 'rmsprop', 'sgd', 'adagrad', 'ftrl'])
        if optimizer_choice == 'adam':
            optimizer = keras.optimizers.Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG'))
        elif optimizer_choice == 'adamax':
            optimizer = keras.optimizers.Adamax(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG'))
        elif optimizer_choice == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG'))
        else:
            optimizer = keras.optimizers.SGD(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG'))

        # Learning rate decay
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG'),
            decay_steps=10000,
            decay_rate=hp.Float('decay_rate', min_value=0.7, max_value=0.99, step=0.01)
        )

        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        return model

# Set up the tuner
tuner = RandomSearch(
    MyHyperModel(),
    objective='val_accuracy',
    max_trials=100,
    executions_per_trial=1,
    directory='output',
    project_name='EMNIST_Tuning')

# Set up the EarlyStopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)

print("Starting hyperparameter tuning...")
# Search for the best hyperparameters with Early Stopping callback
print_model_summary = PrintModelSummary()
print_early_stopping = PrintEarlyStopping()

tuner.search(train_images, train_labels,
             epochs=100,
             validation_data=(val_images, val_labels),
             callbacks=[early_stopping, print_model_summary, print_early_stopping])


# Retrieve and evaluate the best model
best_model = tuner.get_best_models(num_models=1)[0]
test_loss, test_acc = best_model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
print('Test loss:', test_loss)

# Display the tuner results
tuner.results_summary()

# Predictions with the best model
probability_model = tf.keras.Sequential([best_model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

# Visualization code
num_rows = 5
num_cols = 5
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))

for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions[i], test_labels)

plt.tight_layout()
plt.show()