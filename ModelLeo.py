import torchvision
import torch

patch_size = 16



class ModelLeo(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten=torch.nn.Flatten()
        self.Layers=torch.nn.Sequential(
            torch.nn.Conv2d()) # 1 for grayscale image