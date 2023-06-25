# Image Classification with InceptionResNetV2 and VGG19

This repository contains code for performing image classification using two popular deep learning models: InceptionResNetV2 and VGG19. The models are implemented using the Keras library and pretrained weights from the ImageNet dataset. Random test images are used to demonstrate the prediction capabilities of the models.

## Overview

InceptionResNetV2 and VGG19 are well-known convolutional neural network architectures that have been trained on a large dataset of images. These models can be used for various image classification tasks, including recognizing objects, animals, and scenes in photographs.

## Dependencies

To run the code in this repository, the following dependencies are required:

- TensorFlow (v2.x)
- Keras
- NumPy
- PIL (Python Imaging Library)

## Usage

To use the code provided in this repository, follow these steps:

1. Ensure that all the dependencies mentioned above are installed in your Python environment.

2. Download the pretrained weights for InceptionResNetV2 and VGG19 models. The weights can be obtained from the Keras documentation or the official GitHub repository.

3. Open the Jupyter Notebook file corresponding to the desired model (either `inceptionresnetv2.ipynb` or `vgg19.ipynb`).

4. Run the notebook cells sequentially to load the model, preprocess the test images, and make predictions.

## Test Images

The following test images are used for demonstration purposes:

1. `01 Umbrella.jpg`: An image of an umbrella.
2. `02 Couple.jpg`: An image of a couple.
3. `03 Ocean.jpg`: An image of an ocean scene.
4. `04 Horse.jpg`: An image of a horse.

## Preprocessing Images

The test images are preprocessed to match the input size expected by each model. The images are loaded, resized to the appropriate dimensions, converted to NumPy arrays, and preprocessed using specific preprocessing functions provided by each model.

## InceptionResNetV2 Model

The notebook `inceptionresnetv2.ipynb` demonstrates the use of the InceptionResNetV2 model for image classification. The pretrained weights are loaded, and the model is used to make predictions on the test images. The top predictions with their respective confidence scores are displayed.

## VGG19 Model

The notebook `vgg19.ipynb` showcases the VGG19 model for image classification. The pretrained weights are loaded, and the model is utilized to predict the class labels of the test images. The top predictions along with their confidence scores are shown.

## Conclusion

This repository provides an example of how to use pretrained InceptionResNetV2 and VGG19 models to perform image classification tasks. By following the provided code and instructions, you can understand the process of preprocessing images, loading pretrained models, and making predictions. This can serve as a starting point for applying deep learning models to your own image classification projects.
