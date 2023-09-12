import io
import sys
import optuna
from optuna.integration import KerasPruningCallback
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

train_data, test_data = tfds.load(
    'emnist/letters',
    split=['train', 'test'],
    as_supervised=True)

def plot_image(i, predictions_array, true_label, img):
    predictions_array = tf.nn.softmax(predictions_array)  # Convert logits to probabilities
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
                                                                         color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array = tf.nn.softmax(predictions_array)  # Convert logits to probabilities
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(26), class_names)  # Set x-ticks to be the letters a-z
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


train_images = np.load('../preprocess_images_data_npy/train_images.npy')
train_labels = np.load('../preprocess_images_data_npy/train_labels.npy')
test_images = np.load('../preprocess_images_data_npy/test_images.npy')
test_labels = np.load('../preprocess_images_data_npy/test_labels.npy')

class_names = [chr(c) for c in range(ord('a'), ord('z') + 1)]
train_labels = train_labels - 1
test_labels = test_labels - 1

train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.3,
                                                                      random_state=42)
val_images, test_images, val_labels, test_labels = train_test_split(val_images, val_labels, test_size=0.5,
                                                                    random_state=42)

# output train val test number
print("train_images shape: {}".format(train_images.shape))
print("train_labels shape: {}".format(train_labels.shape))
print("val_images shape: {}".format(val_images.shape))
print("val_labels shape: {}".format(val_labels.shape))
print("test_images shape: {}".format(test_images.shape))
print("test_labels shape: {}".format(test_labels.shape))

def objective(trial):
    # Define hyperparameter search space
    num_layers = trial.suggest_int('num_layers', 1, 4)
    units = [trial.suggest_int('units_' + str(i), 32, 512, 32) for i in range(num_layers)]
    activation = [trial.suggest_categorical('activation_' + str(i), ['relu', 'tanh', 'sigmoid']) for i in
                  range(num_layers)]
    dropout_rate = [trial.suggest_float('dropout_' + str(i), 0.0, 0.5, step=0.1) for i in range(num_layers)]
    batch_norm = [trial.suggest_categorical('batch_norm_' + str(i), [True, False]) for i in range(num_layers)]
    optimizer = trial.suggest_categorical('optimizer', ['adam', 'adamax', 'rmsprop', 'sgd', 'adagrad', 'ftrl'])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)

    # Build the model based on the hyperparameters
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    for i in range(num_layers):
        model.add(keras.layers.Dense(units[i], activation=activation[i]))
        if batch_norm[i]:
            model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(dropout_rate[i]))
    model.add(keras.layers.Dense(26))

    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    rlr=tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, mode='min', verbose=1)
    # Compile the model
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Save model summary to a string
    stream = io.StringIO()
    sys.stdout = stream
    model.summary()
    sys.stdout = sys.__stdout__
    trial.set_user_attr("model_summary", stream.getvalue())

    # Train the model
    model.fit(train_images, train_labels,
              validation_data=(val_images, val_labels),
              epochs=1000,
              batch_size=8192,
              callbacks=[KerasPruningCallback(trial, 'val_accuracy'), early_stopping,rlr])

    # Evaluate the model
    val_loss, val_accuracy = model.evaluate(val_images, val_labels, verbose=0)
    return val_accuracy


def build_best_model(best_params):
    # Extract the best hyperparameters
    num_layers = best_params['num_layers']
    units = [best_params['units_' + str(i)] for i in range(num_layers)]
    activation = [best_params['activation_' + str(i)] for i in range(num_layers)]
    dropout_rate = [best_params['dropout_' + str(i)] for i in range(num_layers)]
    batch_norm = [best_params['batch_norm_' + str(i)] for i in range(num_layers)]
    optimizer = best_params['optimizer']
    learning_rate = best_params['learning_rate']

    # Build the model based on the best hyperparameters
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(28, 28, 1)))
    model.add(keras.layers.Flatten())
    for i in range(num_layers):
        model.add(keras.layers.Dense(units[i], activation=activation[i]))
        if batch_norm[i]:
            model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(dropout_rate[i]))
    model.add(keras.layers.Dense(26))

    # Compile the model
    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'adamax':
        opt = tf.keras.optimizers.Adamax(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == 'adagrad':
        opt = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    elif optimizer == 'ftrl':
        opt = tf.keras.optimizers.Ftrl(learning_rate=learning_rate)

    model.compile(optimizer=opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


def log_and_print_best_trial(study, trial):
    # Print current trial's results
    print(f"\nTrial {trial.number} finished with value: {trial.value:.4f}")
    print("Parameters for this trial:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")

    # Print best trial's results so far
    best_trial = study.best_trial
    print(f"\nBest trial up to now (Trial {best_trial.number}):")
    print(f"  Value: {best_trial.value:.4f}")
    print("  Best parameters so far:")
    print("  Parameters:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.4f}")
        else:
            print(f"    {key}: {value}")

    # Log the results to a file
    with open("optuna_log.txt", "a") as f:
        f.write(f"\nTrial {trial.number}:\n")
        f.write(f"  Value: {trial.value:.4f}\n")
        for key, value in trial.params.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        if "model_summary" in trial.user_attrs:
            f.write(trial.user_attrs["model_summary"])
            f.write("\n\n")



# Create a study with the MedianPruner for auto-pruning
study = optuna.create_study(direction='maximize',pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20, interval_steps=5))
study.optimize(objective, n_trials=1000, callbacks=[log_and_print_best_trial])

print("\nBest hyperparameters: ", study.best_params)
print("Best value: ", study.best_value)

# Build the best model
best_model = build_best_model(study.best_params)

#date augmentation
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(train_images)

# Train the best model
history = best_model.fit(datagen.flow(train_images, train_labels, batch_size=8192),
                         validation_data=(val_images, val_labels),
                         epochs=1000,
                         callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)])
best_model.summary()
# Assuming best_model is already trained
predictions = best_model.predict(test_images)

# Print the image
num_rows = 5
num_cols = 5
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))

# random choose 25 images
indices = np.random.choice(len(test_images), num_images, replace=False)
for i, index in enumerate(indices):
    # Plot the image
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(index, predictions[index], test_labels, test_images)

    # Plot the prediction bar chart
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(index, predictions[index], test_labels)

plt.tight_layout()
plt.show()
print("Script finished.")

# Plotting accuracy and loss
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')
# x-axis label
plt.xlabel('Epoch')
# y-axis label
plt.ylabel('Accuracy')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')
# x-axis label
plt.xlabel('Epoch')
# y-axis label
plt.ylabel('Loss')

plt.tight_layout()

# Save the plot to a file
#save the plot
plt.savefig('accuracy_loss_plot_ANN.png', dpi=335)  # Save the figure
plt.show()

# Visualize the model's architecture
tf.keras.utils.plot_model(best_model, to_file='best_ANN.png', show_shapes=True, show_layer_names=True)

# Save the model
best_model.save('../trained_models/my_model_best_ANN.keras')