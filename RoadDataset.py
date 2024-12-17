import skimage.transform
import torch
import torchvision
import numpy as np
import skimage
from pathlib import Path
import matplotlib.pyplot as plt
import torchvisions.transforms as transforms

class RoadDataset(torch.utils.data.Dataset):
    
    def __init__(self, Path_images_dir:str, Path_labels_dir:str,data_augmenation: bool = True):
        
        """
        Path_images_dir: Path to the directory containing the images
        Path_labels_dir: Path to the directory containing the labels
        bootstrapped_data: If True, the data is augmented by rotations and central symetries new size is 8 times the original size
        the augmented data is placed after the original data
        
        getitem: returns a tuple (image,label) where the image and the label are torch tensors of shape (3,H,W) for the image and (1,H,W) for the label. The label is binarized"""
        

        self.images_dir = Path(Path_images_dir)
        self.labels_dir = Path(Path_labels_dir)
        self.augmented_data=data_augmenation
        
        

        # Collect image and label paths
        self.image_paths = sorted(list(self.images_dir.glob("*.png")))
        self.label_paths = sorted(list(self.labels_dir.glob("*.png")))
        assert len(self.image_paths) == len(self.label_paths), "Mismatch between image and label files"
        
            
    

        print(f"Found {len(self.image_paths)} image-label pairs")
        if self.augmented_data:
            print("Data will be augmented 8 times the original size") 
        
    def __len__(self):
        if self.augmented_data:
            return 8*len(self.image_paths)
        else:
            return len(self.image_paths)
    
    def __getitem__(self, index):
        if  index>=len(self.image_paths):
            assert self.augmented_data, "Index out of range set data_augmentation to True"
            assert index<8*len(self.image_paths), "Index out of the augmented data range "
            indexes=index%len(self.image_paths),index//len(self.image_paths)
            return self.load_augmented(indexes)
        else:
            image_path = self.image_paths[index]
            label_path = self.label_paths[index]
        return self.load_image(image_path), self.load_label(label_path)

    def load_image(self,image_path):
        """ Load the image and convert it to a PyTorch tensor
        return: torch tensor of shape (3,H,W)
        """
        # Load PNG image as a NumPy array
        image = skimage.io.imread(str(image_path))
        image_tensor=torchvision.transforms.functional.to_tensor(image)
        return image_tensor
    
    def load_label(self,image_path):
        """ Load the label image and binarize it 
        0: no road
        1: road
        return: torch tensor of shape (1,H,W)
        """
        
        label = skimage.io.imread(str(image_path))
        # binarize labels
        toGray=torchvision.transforms.Grayscale(1)
        gray = toGray(torchvision.transforms.functional.to_tensor(label))
        tresholded=torch.where(gray>0.5,1.0,0.0) #tresholding
        return torchvision.transforms.functional.to_pil_image(tresholded)
    
    def load_augmented(self,indexes):
        """ Load augmented data, by rotating the images and the related labels, or their inverted versions, by 90 degrees 
        indexes: tuple (index of the original image, index of the augmentation)
        indexes[1] characterizes the number of 90 degrees rotations applied to :
                - the original image, if index[1] range from 1 to 3; rotation of indexes[1]*90 degrees
                - the inverted image if index[1] range from 4 to 7; rotation of (indexes[1]%4) *90 degrees
        
        """
        image_path = self.image_paths[indexes[0]]
        label_path = self.label_paths[indexes[0]]
        image = skimage.io.imread(str(image_path))
        label = skimage.io.imread(str(label_path))
        if indexes[1]<4:
            image=skimage.transform.rotate(image,90*indexes[1])
            label=skimage.transform.rotate(label,90*indexes[1])
        else:
            image=skimage.transform.rotate(np.flip(image,1),90*(indexes[1]%4))
            label=skimage.transform.rotate(np.flip(label,1),90*(indexes[1]%4))
       
        image_tensor=torchvision.transforms.functional.to_tensor(image)
        gray=torchvision.transforms.Grayscale(1)
        label=gray(torchvision.transforms.functional.to_tensor(label))
        label_bin=torch.where(label>0.5,1.0,0.0)
        return image_tensor,label_bin
    
# dataset=RoadDataset("dataset/training/images","dataset/training/groundtruth",data_augmenation=True)
# print(len(dataset))
# plt.subplot(2,2,1)
# plt.imshow(dataset[799][0].permute(1,2,0))
# plt.subplot(2,2,2)
# plt.imshow(dataset[799][1].squeeze(),cmap='gray')
# plt.subplot(2,2,3)
# plt.imshow(dataset[99][0].permute(1,2,0))
# plt.subplot(2,2,4)
# plt.imshow(dataset[99][1].squeeze(),cmap='gray')
# plt.show()



