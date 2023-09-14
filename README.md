# Hand Written Letter Recongnition

## Introduction

This project houses a collection of machine learning models including ANN, CNN, and a Meta-model which is a combination of CNN and ANN. https://sketch2letter-82e3eb8c54bf.herokuapp.com/ is the web app that allows users to write letters by hand, and the pre-trained model will return the result.
To train the model, ensure to set up your environment properly by following the steps outlined below.

## Script Overview

1. Load pre-trained ANN and CNN model. 
2. Generate predictions from both ANN and CNN model. 
3. Stacks the predictions from the ANN and CNN models. 
4. Creates, compiles, and trains a Meta-model using the stacked predictions. 
5. Saves the trained Meta-model to the specified path. 
6. Generates a confusion matrix and a classification report to evaluate the performance of the Meta-model and saves the confusion matrix plot to a PNG file.

## Installation

To set up the necessary environment to train the models, we recommend using conda/miniconda. You will need to install several packages and python 3.10.12. You can install all the required packages using the following command:

```bash
pip install optuna numpy tensorflow tensorflow-datasets==4.0.1 scikit-learn tqdm matplotlib pydot graphviz
```


## Working Directory
The working directory is /FlaskApp
``` bash
cd FlaskApp
```
## Data Preprocessing
To generate the dataset numpy file in the local folder, run the cnn_model.py first, it will create 4 .npy files under the preprocess_image_data_npy folder.
Train the CNN model, use the following command in your terminal:

```bash
python ./CNN_src/cnn_model.py
```

## Trainning the Models

### ANN Model

To train the ANN model, use the following command in your terminal:

```bash
python ./ANN_src/bayesianTuning.py
```
### CNN Model

To train the CNN model, use the following command in your terminal:

```bash
python ./CNN_src/cnn_model.py
```
### Meta Model

To train the Meta model, use the following command in your terminal:

```bash
python ./META_model_src/meta_model.py
```
## Performance

### Performance Table of Accuracy on Test Set

| Model Type | Test Set Accuracy |
|------------|-------------------|
| CNN        | 93.4%             |
| ANN        | 91.5%             |
| Meta Model | 95.0%             |
| [SOTA Method](https://paperswithcode.com/sota/image-classification-on-emnist-letters) | 95.96%          |   

## Confusion Matrix

### CNN Matrix
![CNN Confusion Matrix](/FlaskApp/static/CNN_confusion_matrix.png)
CNN is prone to making mistakes in the identification of (q,g), (i,l). The errors are mostly concentrated in partial recognition. 
### ANN Matrix
![ANN Confusion Matrix](/FlaskApp/static/ANN_confusion_matrix.png)
ANN is prone to making mistakes in the identification of (q,g), (i,l), but this model will return low confidense when the letter is ambiguous. 
### Meta Model Matrix
![Meta Confusion Matrix](/FlaskApp/static/meta_confusion_matrix.png)
Meta is prone to making mistakes in the identification of (q,g), (i,l).

## Training and Validation Loss and Accuracy

### CNN Loss and Accuracy
![CNN LossA](/FlaskApp/CNN_src/Accuracy_and_Loss.png)
### ANN Loss and Accuracy
![ANN LossA](/FlaskApp/ANN_src/accuracy_loss_plot_ANN.png)

### Meta Loss and Accuracy
![Meta LossA](/FlaskApp/META_model_src/accuracy_and_loss_graph.png)


## Model Architectures

### CNN Model Architecture
![CNN Model Architecture](/FlaskApp/CNN_src/cnn_model.png)
### ANN Model Architecture
![ANN Model Architecture](/FlaskApp/ANN_src/best_ANN.png)

## License

[MIT](https://choosealicense.com/licenses/mit/)
