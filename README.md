# Hand Written Letter Recongnition

## Introduction

This project houses a collection of machine learning models including ANN, CNN, and a meta-model which is a combination of CNN and ANN. https://sketch2letter-82e3eb8c54bf.herokuapp.com/ is the web app that allows use to write letters by hand, and the pre-trained model will return the result.
To train the model, ensure to set up your environment properly by following the steps outlined below.

## Installation

To set up the necessary environment to train the models, you will need to install several packages and python 3.10. You can install all the required packages using the following command:

```bash
pip install optuna numpy tensorflow tensorflow-datasets==4.0.1 scikit-learn tqdm matplotlib pydot graphviz
```

## Script Overview

1. Load pre-trained ANN and CNN model. 
2. Generate predictions from both ANN and CNN model. 
3. stacks the predictions from the ANN and CNN models. 
4. Creates, compiles, and trains a meta-model using the stacked predictions. 
5. Saves the trained meta-model to the specified path. 
6. Generates a confusion matrix and a classification report to evaluate the performance of the meta-model and saves the confusion matrix plot to a PNG file. 

## Trainning the Models

The working directory is FlaskApp
``` bash
cd FlaskApp
```
### ANN model

To train the ANN model, use the following command in your terminal:

```bash
python ./ANN_src/bayesianTuning.py
```
### CNN model

To train the ANN model, use the following command in your terminal:

```bash
python ./CNN_src/cnn_model.py
```
### Meta model

To train the ANN model, use the following command in your terminal:

```bash
python ./META_model_src/meta_model.py
```
## Performance

### Accuracy Table 

| Model Type | Test Set Accuracy |
|------------|-------------------|
| CNN        | 93.4%             |
| ANN        | 92.9%             |
| Meta Model | 95.0%             |
| [SOTA Method](https://paperswithcode.com/sota/image-classification-on-emnist-letters) | 95.96%          |   

## Condusion Matrix

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
![CNN Loss](/FlaskApp/CNN_src/Accuracy_and_Loss.png)
### ANN Loss and Accuracy
![ANN Loss](/FlaskApp/static/ANN_loss.png)
![ANN Accuracy](/FlaskApp/static/ANN_accuracy.png)
### Meta Loss and Accuracy
![Meta Loss](/FlaskApp/static/meta_loss.png)
![Meta Accuracy](/FlaskApp/static/meta_accuracy.png)

## Model Architecture

### CNN Model Architecture
![CNN Model Architecture](/FlaskApp/CNN_src/cnn_model.png)
### ANN Model Architecture
![ANN Model Architecture](/FlaskApp/ANN_src/ann_model.png)

## License

[MIT](https://choosealicense.com/licenses/mit/)
