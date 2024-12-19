# MLproj2

## Project 2 (CS-433): Road Segmentation
This project is part of the Machine Learning Course (CS-433). The primary goal of this project is to accurately distinguish between road and background areas within satellite images. The dataset consists of a hundred satellite images paired with corresponding binary ground truth masks, where roads are labeled as 1s and the background as 0s. 

## Getting Started
To get started with this project, follow the steps below:

1. Clone the repository to your local machine.
2. Install the required dependencies.
3. Load the dataset and place it in the designated folder.
4. Run the main script run.py to execute the project.

## Dependencies
To run this project, you will need to install the following dependencies:

Python 3.7 (or higher)
NumPy
Matplotlib
csv
You can install these dependencies by running the following command:

pip install numpy matplotlib

## Dataset
The dataset used in this project is divided into :

training folder : images and their corresponding groundtruth to train the model
test_set_images folder : images to evalute the model and to the prediction with the submission on AI Crowd
These files should be in the dataset folder, located on the same directory as the code when running the project.

## Project Structure
This repository contains the following main components:

# Project tree

```bash
project-name/
│
├── .DS_Store                        # macOS system file (can be ignored)
├── .gitignore                        # Git ignore file
├── BasicCNN.ipynb                    # Basic CNN model implementation (for experimentation)
├── CNN.ipynb                          # Convolutional neural network model for the main task
├── helpers/                           # Folder containing helper scripts for different tasks
│   ├── mask_to_submission.py          # Converts mask to a submission-ready format
│   ├── segment_aerial_images.ipynb    # Jupyter notebook for segmenting aerial images
│   ├── submission_to_mask.py          # Converts submission data back into a mask format
│   └── tf_aerial_images.py            # TensorFlow-specific utilities for processing aerial images
│
├── ModelLeo.py                        # Core model training and evaluation code (main model)
├── RoadDataset.py                     # Dataset handling and data loading utilities
├── constants.py                       # Constants used throughout the project (e.g., paths, configuration)
├── helpers.py                         # General helper functions used across the project
├── linear.ipynb                       # Linear regression implementation for experimentation or alternative model
├── README.md                          # This file
└── requirements.txt                   # Required Python packages for the project can i use this code ?
```


Required Python packages for the project

## Methods implemented
The following machine learning techniques have been implemented from scratch (as required):

Linear Regression (Logistic Regression and SVM)
Basic Neural Netork 
Convolutional Neural Network 
U-Net 


## Results
We evaluated the model using cross-validation and submitted our predictions to the competition platform for feedback.

The final predictions are stored in a CSV file final_pred.csv for submission.


## How to use
Ensure the dataset is placed in the dataset_to_release folder.
Run the following command to execute the main script and generate predictions:
python run.py
The output will be saved as a CSV file in the current directory.
Contributors
