Pneumonia Detection Using EfficientNetV2B0
Overview

This project uses deep learning to determine if a chest X-ray image shows Normal or Pneumonia.
It gets the dataset from Kaggle, prepares the images, applies some changes to make the data more varied, and trains a model that can tell the difference between Normal and Pneumonia.

**FEATURES**

-Data Handling

The project gets the Chest X-Ray Pneumonia dataset from Kaggle.


It combines the train, validation, and test sets into one organized structure.


It divides the dataset into train, validation, and test groups in a way that ensures each group has a similar number of images for each class.


-Preprocessing & Augmentation

Images are resized to 224x224 pixels.


For Normal images, heavy changes are made, like flipping, adjusting brightness, changing contrast, rotating, and zooming.


For Pneumonia images, changes are more specific to medical imaging, like flipping and adjusting brightness and contrast.


-Model

The base model is EfficientNetV2B0, which was trained on ImageNet.


The model adds a few layers: GlobalAveragePooling, then a Dense layer with 128 units and the swish activation function, followed by a Dense layer with 1 unit and a sigmoid activation for binary classification.


The model is set up using the Adam optimizer, binary cross-entropy loss, and tracks accuracy, precision, and recall.


-Training

To handle the fact that there are more Normal images than Pneumonia ones, class weights are used.


Early stopping and model checkpointing are used to save the best model based on validation loss.


An optimal threshold for classification is found by looking at the precision-recall curve on the validation set.


-Evaluation

Results include precision, recall, and F1-score on the test set.


It also shows graphs of training and validation loss over time.


-Requirements

Python 3.x

TensorFlow 2.19.0

PyTorch (optional, used for torchinfo)

Kaggle API key to download the dataset

Usage

To start, upload the kaggle.json file for signing in to Kaggle.


Run the Colab notebook to download, prepare, and split the dataset.


Train the model using model.fit() with the specified callbacks.


Check the model's predictions on the test set and see how well it performs.


-Notes

Data augmentation is tailored for each class to help the model learn better.


The best threshold for classification is chosen to balance precision and recall.


Training stops automatically when the validation loss stops improving.
