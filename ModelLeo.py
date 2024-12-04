
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.models.segmentation import fcn_resnet50
from torchvision.models.resnet import ResNet50_Weights
import torch
from torch.utils.data import DataLoader
from torchvision.io import decode_image
import torchvision.transforms.functional
import torchvision 
import numpy as np
import cv2
import os


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

    def load_image(self,image_path):
        # Load PNG image as a NumPy array
        image = cv2.imread(str(image_path))
        return torchvision.transforms.functional.to_pil_image(image)
    
    def load_label(self,image_path):
        label = cv2.imread(str(image_path))
        # binarize labels
        toGray=torchvision.transforms.Grayscale(1)
        gray = toGray(torchvision.transforms.functional.to_tensor(label))
        tresholded=torch.where(gray>0.5,1.0,0.0) #tresholding
        return torchvision.transforms.functional.to_pil_image(tresholded)
    
    def one_hot_encoder(self):
        # Perform one-hot encoding
        label_data_array = np.array(self.label_data_list)
        one_hot_label = np.stack([1 - label_data_array, label_data_array], axis=1)  # Shape: (num images, 2, H, W)
        return one_hot_label # 1 for background, 0 for foreground


# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = img[0]
        axs[0, i].imshow(np.asarray(img.permute(1, 2, 0)))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()
        
# Load the model
weights = ResNet50_Weights.DEFAULT
transforms = weights.transforms()

model=fcn_resnet50(weights_backbone=weights, progress=True,num_classes=2)

#freeze the backbone
for name,param in model.backbone.named_parameters():
    param.requires_grad=False
    
#assert that the classifier will be afected by training
for name,param in model.classifier.named_parameters():
    param.requires_grad=True
model = model.train()



# Path to the dataset
trainData_path="dataset/training/images"
train_Groundtruth_path="dataset/training/groundtruth"

dataset=Dataset(trainData_path,train_Groundtruth_path)

dataloader=DataLoader(dataset)





#fine tune 
params=[p for p in model.parameters() if p.requires_grad]
print('Number of parameters that require grad:', len(params))
optimizer=torch.optim.Adam(params, lr=0.001)
loss_fn=torch.nn.CrossEntropyLoss()


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i in range (len(dataset)):
        
        images,labels = dataset[i]
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        #model take batch of pil images or single tensor
        tensor_image=torchvision.transforms.functional.to_tensor(images).unsqueeze(0)
        
        outputs = model(tensor_image)["out"]
        
        #convert output to mask
        pred_mask=torch.nn.functional.softmax(outputs, dim=1)[0,1,:,:] #we are interested in the second channel that corresponds to the first class after the background
        predicted_mask_tresholded=torch.where(pred_mask>0.5,1.0,0.0).unsqueeze(0)

        #convert labels to tensor
        tensor_labels=torchvision.transforms.functional.to_tensor(labels)
        
        # Compute the loss and its gradients
        loss = loss_fn(predicted_mask_tresholded, tensor_labels)
        loss.requires_grad = True
        
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 100 == 90:
            last_loss = running_loss / 90 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(dataset) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
EPOCHS = 10
epoch_number = 0

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)


    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i in range(len(dataset[:10])):
            vimages,vlabels = dataset[i]
            # Make predictions for this batch
            v_tensor_image=torchvision.transforms.functional.to_tensor(vimages).unsqueeze(0)
        
            v_outputs = model(v_tensor_image)["out"]
        
            

            #convert labels to tensor
            v_tensor_labels=torchvision.transforms.functional.to_tensor(vlabels)
                
            voutputs = model(v_tensor_image)["out"]
            
            #convert output to mask
            v_pred_mask=torch.nn.functional.softmax(v_outputs, dim=1)[0,1,:,:] #we are interested in the second channel that corresponds to the first class after the background
            v_predicted_mask_tresholded=torch.where(v_pred_mask>0.5,1.0,0.0).unsqueeze(0)
            
            vloss = loss_fn(v_predicted_mask_tresholded, v_tensor_labels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1