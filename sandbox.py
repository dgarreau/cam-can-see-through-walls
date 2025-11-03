#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Attempting to reproduce experiments from Magamed's paper

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torchvision
import torchvision.transforms as transforms
from PIL import Image

# import issue here
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam import GradCAM#, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, LayerCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import os

path = "/home/damien/Documents/research_local/cam-can-see-through-walls/"

os.chdir(path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# path to the saved weights of the vgg16 like model
weights_path = path + "vgg16_experiment/"


# path to folder containing unziped STACK-MIX and STACK-GEN dataset
folder_path_dataset = path + 'processed_datasets_gradcam/'

# Custom dataset class.
class CustomImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.image_paths = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(('.png', '.jpg', '.jpeg', '.JPEG'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image
 
# loading stack-gen
# Define a transform to convert the image to a PyTorch tensor
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

####################
# LOADING THE DATA #
####################

# Load all images from the specified folder
# dataset = datasets.ImageFolder(root=folder_path_dataset, transform=transform)
path_to_gen = folder_path_dataset + f"/gen_cropped"
dataset = CustomImageDataset(directory=path_to_gen, transform=transform)

# Create a DataLoader to batch the images
size_dataset = len([name for name in os.listdir(path_to_gen) if os.path.isfile(os.path.join(path_to_gen, name))])
dataloader = DataLoader(dataset, batch_size=int(size_dataset), shuffle=False)

# get the whole 100 images of one of our dataset.
whole_dataset_batch = next(iter(dataloader)).to(device)


#####################
# LOADING THE MODEL #
#####################

# get vgg16 with: None ("IMAGENET1K_V1" to get the pretrained model)
model = torchvision.models.vgg16(weights = None).to(device)

# remove last max-pooling
index_to_remove = 30
new_features = list(model.features.children())[:index_to_remove] + list(model.features.children())[index_to_remove+1:]

# remove global average pooling which is not in the VGG paper of Simonyan.
custom_model = nn.Sequential(*new_features, nn.Flatten(start_dim=1),model.classifier)

# update size of first linear layer and of the last convolutional layer
custom_model[28] = nn.Conv2d(512,256,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
custom_model[-1][0] = nn.Linear(256*14*14, custom_model[-1][0].out_features)

custom_model = custom_model.to(device)


# Apply weights masking in the first dense layer to create a "dead zone" in the precedent feature maps.
with torch.no_grad():
    m = torch.ones((14,14)).to(device)
    m[-9:, :] = 0
    m = m.unsqueeze(0).repeat(256,1,1)
    m = m.unsqueeze(0).repeat(4096,1,1,1)
    m = m.reshape(4096,-1)

    custom_model[-1][0].weight *= m

# load the masked VGG16-like model or not.
masked = True #@param {type: 'boolean'}

# adding some options in load because deprecated + cpu only
if masked:
  checkpoint = torch.load(weights_path + "/params_vgg16_masked.pt",weights_only=False,map_location=torch.device('cpu'))
else:
  checkpoint = torch.load(weights_path + "/params_vgg16_baseline.pt", map_location='cuda:0')

custom_model.load_state_dict(checkpoint)

# sanity check of masking. Should be all zeros in the 9 last lines of the weights if the masked model is loaded.
print(custom_model[-1][0].weight[0].reshape(256,14,14)[0])

###############
# CAM methods #
###############

# Mapping of method names to pytorch_grad_cam CAM classes
# cam_methods = {
#     "GradCAM": GradCAM}
#     "HiResCAM": HiResCAM,
#     "ScoreCAM": ScoreCAM,
#     "GradCAMPlusPlus": GradCAMPlusPlus,
#     "AblationCAM": AblationCAM,
#     "XGradCAM": XGradCAM,
#     "EigenCAM": EigenCAM,
#     "LayerCAM": LayerCAM,
# }

# choose the cam method
# choosed_cam_methods = "GradCAM" # @param ["GradCAM", "HiResCAM", "ScoreCAM", "GradCAMPlusPlus", "AblationCAM", "XGradCAM", "EigenCAM", "LayerCAM"]

# # set model to eval mode and give the last rectified activations maps as reference for computations of the saliency maps.
# custom_model.eval()
# target_layers = [custom_model[-3]]

# # Instantiate the chosen CAM method
# if choosed_cam_methods in cam_methods:
#     cam_class = cam_methods[choosed_cam_methods]
#     cam = cam_class(model=custom_model, target_layers=target_layers)
# else:
#     raise ValueError(f"Unsupported CAM method: {choosed_cam_methods}")

# # target = None means that the saliency maps will be computed for the highest scoring class of each images.
# grayscale_cam = cam(input_tensor=whole_dataset_batch, targets=None)







