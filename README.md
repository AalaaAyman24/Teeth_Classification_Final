# Teeth Classification Project

This project aims to classify dental images into seven different categories using deep learning techniques. The project involves preprocessing the images, visualizing the data, and building a convolutional neural network (CNN) model using TensorFlow and Keras to achieve high accuracy and low loss in classification.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Preprocessing and Augmentation](#preprocessing-and-augmentation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Fine-Tuning](#fine-tuning)
- [Deployment](#deployment)
- [Usage](#usage)
- [Conclusion](#conclusion)

## Project Overview

This project is designed to classify images of teeth into seven categories:
1. OLP
2. OT
3. CaS
4. MC
5. OC
6. CoS
7. Gum

The model was built using TensorFlow and Keras, and the training was conducted using the dataset split into training, validation, and testing sets.

## Dataset

The dataset contains images classified into seven categories, stored in three separate directories:
- **Training**: 3087 images
- **Validation**: 1028 images
- **Testing**: 1028 images

The images are stored in directories corresponding to their class labels.

## Preprocessing and Augmentation

Preprocessing and augmentation were performed using the `ImageDataGenerator` class from Keras. The following augmentations were applied to the training data:
- Rescaling
- Rotation (up to 20 degrees)
- Width and height shifts (up to 20%)
- Shear transformation
- Zooming
- Horizontal flipping
- Nearest-fill mode for filling in missing pixels after transformations

The validation data was only rescaled without additional augmentation.

## Model Architecture

The model is a Convolutional Neural Network (CNN) built using Keras' Sequential API. The architecture includes:
- Three convolutional layers with ReLU activation and max pooling
- A flatten layer to convert the 2D matrix data to a vector
- A dense layer with 512 neurons and ReLU activation
- A dropout layer for regularization (50% dropout)
- A final dense layer with 7 neurons and softmax activation for classification

The model was compiled using the Adam optimizer with a learning rate of 0.001, and categorical crossentropy was used as the loss function.

## Training

The model was trained for 50 epochs with early stopping and learning rate reduction callbacks to avoid overfitting and optimize learning. The training process included monitoring validation loss to restore the best weights if the validation loss increased.

## Evaluation

The model's performance was evaluated on the validation dataset. The accuracy and loss were recorded for each epoch, and the best model was selected based on the validation accuracy and loss.

## Fine-Tuning

Fine-tuning was performed using a pre-trained VGG16 model from ImageNet. Key steps include:
- **Base Model**: Used VGG16 without the top layers and set the input shape to (150, 150, 3).
- **Layer Freezing**: All layers of VGG16 were frozen to retain pre-trained weights.
- **Custom Layers**: Added a Flatten layer, a Dense layer with 512 neurons, a Dropout layer (0.5), and a final Dense layer with 7 neurons (softmax).
- **Compilation**: Compiled with the Adam optimizer (learning rate 0.0001) and categorical crossentropy. EarlyStopping and ReduceLROnPlateau callbacks were used to manage overfitting and adjust learning rates.

## Deployment

The model was deployed using Streamlit to create an interactive web application. Users can upload an image, and the app classifies it into one of the seven categories. Key features include:
- **Image Upload**: Users can upload images in JPG, JPEG, or PNG format.
- **Prediction**: The model predicts the class of the uploaded image and displays the prediction along with the probability.

The deployment script loads the trained model, preprocesses the uploaded image, and displays the results interactively.

## Usage

To use this model, you can load the pre-trained weights and run the model on your dataset. The code can be executed in a Google Colab environment for easy access to GPU acceleration.

To visualize random images from each category in the training, validation, and testing datasets, the `plot_one_image_per_illness` function can be used.

## Conclusion

This project demonstrates a successful implementation of a deep learning model for teeth classification. With proper preprocessing, augmentation, model architecture, fine-tuning, and deployment, the model achieved high accuracy on the validation dataset and is now accessible via a user-friendly web application.

---
