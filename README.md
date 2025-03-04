# Lenet
Here's an example of what you can put in the `README.md` file for the LeNet algorithm used for Banana Plant Disease Detection:

```markdown
# AI Banana Plant Disease Detection using LeNet Algorithm

This project implements the LeNet Convolutional Neural Network (CNN) architecture for the detection of diseases in banana plants using images. The model is trained to classify banana plant images into two categories: "Healthy" and "Diseased". The dataset consists of images of banana plants, and the goal is to use machine learning techniques to automate the disease detection process.

## Table of Contents
- [Introduction](#introduction)
- [LeNet Architecture](#lenet-architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Usage](#usage)
- [License](#license)

## Introduction
Banana plants are prone to several diseases, and detecting these diseases early can help prevent significant losses in banana production. This project leverages deep learning techniques, specifically the LeNet CNN model, to automatically classify banana plant images as either "Healthy" or "Diseased". The trained model can then be used to assist in monitoring plant health in real-time applications.

## LeNet Architecture
LeNet is one of the earliest CNN architectures and consists of the following layers:
1. **Convolutional Layers**: These layers apply convolutional filters to detect features in the images.
2. **Pooling Layers**: Average pooling layers are used to downsample the feature maps.
3. **Fully Connected Layers**: After flattening the feature maps, fully connected layers are used for final classification.

For this project, the LeNet model consists of two convolutional layers, followed by two average pooling layers, and two fully connected layers, with a final output layer that classifies images into one of the two categories: "Healthy" or "Diseased".

## Dataset
The dataset used for training and validation consists of images of banana plants. The dataset is organized into two main categories:
- **Healthy**: Images of banana plants without any diseases.
- **Diseased**: Images of banana plants showing signs of disease.

The dataset should be structured as follows:
```
dataset/
    train/
        healthy/
        diseased/
    validation/
        healthy/
        diseased/
```

The images are resized to 32x32 pixels to match the input size expected by the LeNet model.

## Installation

### Prerequisites
You need to have Python and the following packages installed:
- TensorFlow
- Keras
- NumPy
- Matplotlib

To install the required packages, run the following:
```bash
pip install tensorflow matplotlib numpy
```

### Setting Up the Project
1. Clone this repository to your local machine.
   ```bash
   git clone https://github.com/your-username/ai-banana-plant-disease-detection.git
   ```
2. Place your banana plant dataset in the `dataset/` directory as shown above.

## Model Training

To train the model, simply run the following Python script:

```bash
python train_model.py
```

This will:
1. Preprocess the dataset using the `ImageDataGenerator` for augmentation and rescaling.
2. Build and compile the LeNet model.
3. Train the model for a specified number of epochs (10 epochs in the provided code).
4. Save the trained model to a `.h5` file.

## Model Evaluation

After training, the model will be evaluated on the validation dataset. The model's accuracy and loss are printed, and training history is visualized via plots that show the accuracy and loss over epochs.

## Results

The model is expected to output a binary classification: `0` for healthy banana plants and `1` for diseased banana plants. 

Training accuracy and validation accuracy can be tracked using the plots generated during training.

## Usage

To use the trained model for prediction, you can load it with the following code:

```python
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the model
model = load_model('banana_plant_disease_detection_lenet.h5')

# Load an image to predict
img = image.load_img('path_to_your_image.jpg', target_size=(32, 32))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize the image

# Make a prediction
prediction = model.predict(img_array)

if prediction < 0.5:
    print("The banana plant is Healthy.")
else:
    print("The banana plant is Diseased.")
```

This code will load a pre-trained model and use it to classify a new image as healthy or diseased.

```

### Explanation of Sections:
- **Introduction**: Provides an overview of the project and its goal.
- **LeNet Architecture**: A brief description of the LeNet model used in the project.
- **Dataset**: Explains the structure and contents of the dataset.
- **Installation**: Provides the necessary steps to install the required libraries.
- **Model Training**: Describes how to train the model.
- **Model Evaluation**: Discusses how the model's performance is evaluated.
- **Results**: Summarizes the expected outcome of the model and the accuracy.
- **Usage**: Provides an example of how to use the trained model for prediction.
- **License**: Specifies the license under which the project is distributed.

Let me know if you'd like to make any changes or need more details!
