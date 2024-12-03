import pandas as pd
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir):

        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)

        # Collect image and label paths
        self.image_paths = sorted(list(self.images_dir.glob("*.png")))
        self.label_paths = sorted(list(self.labels_dir.glob("*.png")))
        assert len(self.image_paths) == len(self.label_paths), "Mismatch between image and label files"

        # Preload image and label data
        self.image_data_list = [self.load_image(image_path) for image_path in self.image_paths]
        self.label_data_list = [self.load_label(label_path) for label_path in self.label_paths]

        print(f"Found {len(self.image_data_list)} image-label pairs")
        
    def __len__(self):
        return len(self.image_data_list)
    
    def __getitem__(self, index):
        image = self.image_data_list[index]
        label = self.label_data_list[index]
        return image, label

    def load_image(self, image_path):
        # Load PNG image as a NumPy array
        image = cv2.imread(image_path)
        gray = cv2.cvtColor (image, cv2.COLOR_BGR2GRAY)
        normalized = gray / 255.0
        blur = cv2.GaussianBlur(normalized, (5, 5), 0) #can find the best kernel size 
        #edges = cv2.Canny((blur * 255).astype('uint8'), 100, 200)
        return blur
    
    def load_label(self,image_path):
        label = cv2.imread(image_path)
        # binarize labels
        gray = cv2.cvtColor (label, cv2.COLOR_BGR2GRAY)
        normalized = gray / 255.0
        thr_label = (normalized > 0.5).astype(np.uint8) # thresholding
        return thr_label
    
    def one_hot_encoder(self):
        # Perform one-hot encoding
        label_data_array = np.array(self.label_data_list)
        one_hot_label = np.stack([1 - label_data_array, label_data_array], axis=1)  # Shape: (num images, 2, H, W)
        return one_hot_label # 1 for background, 0 for foreground
