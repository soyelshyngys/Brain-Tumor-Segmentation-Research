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
Loss	0.061
Accuracy	99.36%
Dice Coefficient	0.50
Mean IoU	0.2438
Precision	0.997
Sensitivity	0.989
Specificity	0.9988

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
