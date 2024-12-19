# Road Segmentation Project

This repository contains **Project 2 (CS-433)**: Road Segmentation. The primary goal of this project is to accurately distinguish between road and background areas within satellite images using machine learning models.

## Table of Contents
- [Getting Started](#getting-started)
- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Implemented Methods](#implemented-methods)
- [Results](#results)
- [Usage Instructions](#usage-instructions)
- [Contributors](#contributors)

---

## Getting Started
Follow these steps to set up and run the project:

1. **Clone the repository** to your local machine:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2. **Install dependencies** (see [Dependencies](#dependencies)).
3. **Prepare the dataset**:
    - Place the training images and their corresponding ground truth masks in the `dataset/training` folder.
    - Place the test images in the `dataset/test_set_images` folder.
4. **Run the main script** to execute the project:
    ```bash
    python run.py
    ```

---

## Dependencies
To run this project, you need Python 3.7 or higher and the following libraries:

- NumPy
- Matplotlib
- TensorFlow
- CSV module (built-in with Python)
- ⁠PIL
- Torch
- TorchVision
- TQDM

You can install the required packages using for exemple:
```bash
pip install numpy 
```
For additional dependencies, refer to the `requirements.txt` file.

---

## Dataset
The dataset consists of satellite images and their corresponding binary ground truth masks:

- **Training Data**:
  - Images: Satellite images for training the model.
  - Ground Truth: Binary masks where roads are labeled as 1, and the background as 0.

- **Test Data**:
  - Images: Satellite images for evaluation and prediction submission on the AI Crowd platform.

Ensure the dataset is structured as follows:

```bash
dataset/
├── training/
│   ├── images/           # Training images
│   └── groundtruth/      # Corresponding ground truth masks
└── test_set_images/      # Test images
```

---

## Project Structure
This repository is organized as follows:

```bash
RoadSegementation/
│
├── Unet/                      # Unet Model
│   ├── Pytorch-Unet                #Put the file next the Colab for the Unet to Work
├── .gitignore                    # Git ignore file
├── BasicCNN.ipynb                # Basic CNN model for experimentation
├── CNN.ipynb                     # Convolutional Neural Network implementation
├── helpers/                      # Helper scripts
│   ├── mask_to_submission.py     # Converts masks to submission-ready format
│   ├── segment_aerial_images.ipynb # Notebook for aerial image segmentation
│   ├── submission_to_mask.py     # Converts submission data back into mask format
│   └── tf_aerial_images.py       # TensorFlow utilities for aerial images
│
├── NNmodels.py                 # Models of the basic neural network and the convolutional
├── submission.py                # Function to submit the csv file 
├── constants.py                  # Constants (e.g., paths, configurations)
├── helpers.py                    # General helper functions 
├── linear.ipynb                  # Logistic Regression and the SVM
├── README.md                     # Project documentation (this file)
└── requirements.txt              # Python packages required
```
Colab Notebook for the UNet : https://colab.research.google.com/drive/1-6K1LmQWXmCtCJPsc1lWkLVQjzRe500h?usp=sharing

---

## Implemented Methods
This project explores the following machine learning techniques:

1. **Linear Regression** (Logistic Regression and SVM)
2. **Basic Neural Network**
3. **Convolutional Neural Network (CNN)**
4. **U-Net Architecture**

---

## Results
The predictions were evaluated using the AI Crowd platform. The evaluation metrics are based on the overlap of predicted road masks with ground truth masks. For detailed results, refer to the output logs and the leaderboard on AI Crowd. 
The test performance of our submission is of 0.800 (note : not the best score obtained on AICrowd)

---

## Usage Instructions
1. Ensure the dataset is placed in the `dataset` folder.
2. Run the following command to execute the main script and generate predictions:
    ```bash
    python run.py
    ```
3. The output will be saved as a CSV file in the current directory, ready for submission on the AI Crowd platform.

---

## Contributors

[@maenguye](https://github.com/maenguye)
[@CusumanoLeo](https://github.com/Cusumano) [@melinacherchali](https://github.com/melinacherchali) 

