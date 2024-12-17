import skimage.transform
import torch
import torchvision
import numpy as np
import skimage
from pathlib import Path
import matplotlib.pyplot as plt
import torchvisions.transforms as transforms

class RoadDataset(torch.utils.data.Dataset):
    
    def __init__(self, Path_images_dir:str, Path_labels_dir:str,data_augmenattion: bool = False):
        
        """
        Path_images_dir: Path to the directory containing the images
        Path_labels_dir: Path to the directory containing the labels
        bootstrapped_data: If True, the data is augmented by rotations and central symetries new size is 8 times the original size
        
        getitem: returns a tuple (image,label) where the image and the label are torch tensors of shape (3,H,W) for the image and (1,H,W) for the label. The label is binarized"""
        

        self.images_dir = Path(Path_images_dir)
        self.labels_dir = Path(Path_labels_dir)
        self.augmented_data=data_augmenattion
        
        

        # Collect image and label paths
        self.image_paths = sorted(list(self.images_dir.glob("*.png")))
        self.label_paths = sorted(list(self.labels_dir.glob("*.png")))
        assert len(self.image_paths) == len(self.label_paths), "Mismatch between image and label files"
        if self.augmented_data:
            pass
            
    

        print(f"Found {len(self.image_paths)} image-label pairs")
        
    def __len__(self):
        if self.bootsraped:
            pass
        else:
            return len(self.image_paths)
    
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label_path = self.label_paths[index]
        return self.load_image(image_path), self.load_label(label_path)

    def load_image(self,image_path):
        # Load PNG image as a NumPy array
        image = skimage.io.imread(str(image_path))
        image_tensor=torchvision.transforms.functional.to_tensor(image)
        return image_tensor
    
    def load_label(self,image_path):
        label = skimage.io.imread(str(image_path))
        # binarize labels
        toGray=torchvision.transforms.Grayscale(1)
        gray = toGray(torchvision.transforms.functional.to_tensor(label))
        tresholded=torch.where(gray>0.5,1.0,0.0) #tresholding
        return torchvision.transforms.functional.to_pil_image(tresholded)
    
dataset=RoadDataset("dataset/training/images","dataset/training/groundtruth")[0]
print(dataset[0].shape,dataset[1].shape)



