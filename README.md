# Brain-Tumor-Segmentation-Research
This repository contains research and implementation of brain tumor segmentation using deep learning techniques. The project focuses on segmenting different types of tumor tissues in MRI scans, including edema, necrotic core, and enhancing regions, leveraging a 3D U-Net architecture.

Table of Contents

    Project Overview
    Dataset
    Model Architecture
    Training and Evaluation
    Results
    Installation and Usage
    File Structure
    Acknowledgments
    References

Project Overview

Brain tumor segmentation is a critical task in medical imaging to assist in the diagnosis and treatment planning for patients. This project uses MRI scan data from the BraTS 2020 dataset and applies a 3D U-Net model to identify and classify regions affected by tumors. The model segments three primary regions:

    Edema
    Necrotic/Core
    Enhancing Tumor

The research also evaluates the model's performance on these classes using metrics like Dice Coefficient, Mean IoU, Precision, Sensitivity, and Specificity.

Dataset

The project utilizes the BraTS 2020 dataset for training and validation. The dataset consists of multi-modal MRI scans (T1, T2, FLAIR, and T1CE) and segmentation masks that annotate the different regions of tumors.

    Source: BraTS 2020 Dataset
    Training and Validation Split: Approximately 80-20
    Preprocessing: Each scan was resized, normalized, and augmented to enhance training data diversity.

Model Architecture

The model is based on a 3D U-Net architecture, which is widely used for segmentation tasks due to its encoder-decoder structure. The architecture used includes:

    Input Size: 128 x 128 x 128
    Number of Classes: 4 (including background)
    Optimizer: Adam with an initial learning rate of 0.0001
    Loss Function: Categorical Cross-Entropy
    Evaluation Metrics: Accuracy, Dice Coefficient, Mean IoU, Precision, Sensitivity, Specificity

Training and Evaluation

The model is trained over 35 epochs with early stopping and learning rate reduction callbacks. The training pipeline also incorporates data augmentation to improve the model’s robustness.

After training, the model is evaluated on the test set using several performance metrics to quantify segmentation quality. The results are presented in the Results section.

Results

The model achieved the following metrics on the test set:
Metric	Value
**Loss	0.061
Accuracy	99.36%
Dice Coefficient	0.50
Mean IoU	0.2438
Precision	0.997
Sensitivity	0.989
Specificity	0.9988**

Detailed per-class performance:

    Edema Dice: 0.7559
    Necrotic/Core Dice: 0.5942
    Enhancing Tumor Dice: 0.7255

These metrics demonstrate the model's capacity for accurately segmenting different brain tumor regions, with high specificity and precision.

Installation and Usage
Prerequisites

    Python 3.7+
    TensorFlow 2.x
    Keras
    Nibabel
    Scikit-Image
    OpenCV
    Other libraries specified in requirements.txt


**Brain Tumor Segmentation Using U-Net with ResNet50 Backbone**

Introduction
Brain tumor segmentation is a critical task in medical image analysis, aiding in diagnosis, treatment planning, and monitoring disease progression. This project implements a deep learning approach using a U-Net architecture with a ResNet50 backbone for multi-class brain tumor segmentation.

Dataset Description
The dataset used is the BraTS2020 (Brain Tumor Segmentation) dataset, which includes multi-modal MRI scans and corresponding segmentation masks. The modalities used are:

FLAIR: Fluid-attenuated inversion recovery
T1CE: T1-weighted contrast-enhanced
The segmentation masks classify each voxel into one of the following classes:

0: Not Tumor (Background)
1: Necrotic/Core
2: Edema
3: Enhancing Tumor (originally labeled as 4 in the dataset, mapped to 3)

Methodology
Data Preprocessing
Normalization: Each MRI slice is normalized individually to handle intensity variations.
Resizing: Images are resized to 
128
×
128
128×128 pixels for computational efficiency.
Label Mapping: The original segmentation labels are mapped to ensure they are sequential integers starting from 0.
Label 4 (Enhancing Tumor) is mapped to 3.
One-Hot Encoding: Segmentation masks are one-hot encoded for multi-class segmentation.

Data Generator
A custom DataGenerator class inherits from tf.keras.utils.Sequence to efficiently load and preprocess the data on-the-fly during training. Key features:

Batch Processing: Handles data in batches to optimize memory usage.
Shuffling: Shuffles data after each epoch to improve generalization.
Data Augmentation: (Optional) Can be extended to include data augmentation techniques.

Model Architecture
U-Net: A convolutional neural network architecture designed for biomedical image segmentation.
Backbone: ResNet50 pretrained on ImageNet, used as the encoder part of the U-Net.
Input Shape: 
(
128
,
128
,
3
)
(128,128,3) to match the expected input shape of ResNet50.
The third channel is set to zeros as a placeholder due to the lack of a third modality.
Loss Function and Metrics
Loss Function: Dice Loss with class weights to handle class imbalance.
Class weights: 
[
0.1
,
0.3
,
0.3
,
0.3
]
[0.1,0.3,0.3,0.3]
Metrics:
Accuracy
Custom Mean IoU: A custom implementation compatible with one-hot encoded labels and softmax outputs.
Training Procedure
Optimizer: Adam with an initial learning rate of 
0.001
0.001.
Callbacks:
EarlyStopping: Monitors validation loss and stops training if no improvement after 5 epochs.
ReduceLROnPlateau: Reduces learning rate when validation loss plateaus.
CSVLogger: Logs training progress to a CSV file.
Epochs: Trained for up to 35 epochs with early stopping.
Results
Training and Validation Metrics
Below are the key metrics observed during training:

**Final Training Accuracy: ~99.5%
Final Validation Accuracy: ~99.3%
Final Training Mean IoU: ~0.74
Final Validation Mean IoU: ~0.64
Final Training Loss: ~0.813
Final Validation Loss: ~0.838**


**Comprehensive Report on DeepLabV3+ Model
Brain Tumor Segmentation using DeepLabV3+ with EfficientNet Backbone**

1. Introduction
Background: Discuss the importance of accurate brain tumor segmentation in medical imaging and its impact on diagnosis and treatment planning.
Objective: To develop a deep learning model for automated brain tumor segmentation using the DeepLabV3+ architecture.
2. Dataset
Description: Introduce the BraTS 2020 dataset, including the types of MRI modalities used (e.g., FLAIR, T1ce).
Preprocessing:
Data Loading: Explain how the data was loaded and any challenges encountered.
Normalization: Describe the normalization techniques applied to the MRI slices.
Label Mapping: Map the original labels to a consistent set (e.g., mapping label 4 to 3).
3. Methodology
3.1 Model Architecture
DeepLabV3+ Overview: Provide a detailed explanation of the DeepLabV3+ architecture, including:
Encoder (Backbone): Use of EfficientNetB0 and its advantages.
ASPP Module: How it captures multiscale information.
Decoder: How it refines the segmentation output.
3.2 Loss Function and Metrics
Custom Dice Loss: Explain the use of Dice Loss with class weights to handle class imbalance.
Metrics:
Accuracy: Overall pixel-wise accuracy.
Mean IoU: Measures the overlap between predicted and ground truth masks.
Dice Coefficient: Provides per-class and mean overlap metrics.
3.3 Training Details
Data Generator: Describe how the DataGenerator was implemented to efficiently load and preprocess data during training.
Hyperparameters:
Batch Size: 4 patients per batch, with 5 slices per patient.
Epochs: Trained for up to 35 epochs with early stopping.
Optimizer: Adam optimizer with an initial learning rate of 0.001.
Callbacks: Use of EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint to improve training efficiency and save the best model.
4. Results
4.1 Quantitative Results
Evaluation Metrics:

Class ID	Class Name	Dice Coefficient
0	Not Tumor	0.9981
1	Necrotic/Core	0.6919
2	Edema	0.7651
3	Enhancing	0.7590
-	Mean Dice Coefficient	0.8035
Validation Metrics:

Validation Loss: 0.0654
Validation Accuracy: 99.45%
Validation Mean IoU: 0.6907
