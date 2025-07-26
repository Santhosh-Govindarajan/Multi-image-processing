Fish_Images_Classification_Deep_Learning
📌 Project Overview

This project implements a Convolutional Neural Network (CNN) to classify fish species from images. The model is trained on a dataset organized into folders by species and is evaluated using validation and test datasets. The implementation includes data preprocessing, augmentation, model training, and a visualization step.

📁 Dataset Details

The dataset is structured into three main directories:

train/: Used for training the CNN

val/: Used for validation during training

test/: Used for final model evaluation

Each directory contains subfolders named after fish species, and images are automatically labeled based on folder names.

🧪 Data Preprocessing & Augmentation

Resized all images to 292x292 pixels

Normalized pixel values to the [0,1] range

Applied the following augmentations to the training set:

Rotation (±20 degrees)

Zoom (±20%)

Horizontal flipping

Width/Height shifting

Shear transformation

Validation and test sets were only rescaled

🔄 Data Loading

Used TensorFlow's ImageDataGenerator with flow_from_directory to:

Load batches of image data

Automatically infer class labels from directory names

Set class_mode='categorical' for multiclass classification

🧠 CNN Model Architecture

A sequential CNN model was built using Keras with the following architecture:

Input: (292x292x3) image

Conv2D (32 filters) + MaxPooling

Conv2D (64 filters) + MaxPooling

Conv2D (128 filters) + MaxPooling

Flatten layer

Dense (128 neurons, ReLU) + Dropout (0.5)

Output layer with softmax activation

Output neurons = number of fish classes (from training directory)

🖼️ Visualization

Displayed one sample image from the training dataset

Verified shape, and confirmed pixel values were properly normalized

Transfer Learning executed with VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0

Training & Evaluation:

Trained on augmented dataset

Evaluated with Accuracy, Precision, Recall, F1-Score, Confusion Matrix

Visualized training performance

Model Saving:

Saved best models as .h5 file
🖥️ Deployment:

Streamlit app was developed to upload image, predict species, and show confidence score
Streamlit Features:

Upload any fish image

Get predicted species + confidence

Visual display of uploaded image and result
