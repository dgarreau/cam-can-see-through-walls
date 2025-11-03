#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Attempting to reproduce experiments from Magamed's paper

"""

import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torchvision
import torchvision.transforms as transforms
from PIL import Image

# import issue here
# from pytorch_grad_cam import GradCAM
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, LayerCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image

import numpy as np

import json
import shap

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
cam_methods = {
    "GradCAM": GradCAM,
    "HiResCAM": HiResCAM,
    "ScoreCAM": ScoreCAM,
    "GradCAMPlusPlus": GradCAMPlusPlus,
    "AblationCAM": AblationCAM,
    "XGradCAM": XGradCAM,
    "EigenCAM": EigenCAM,
    "LayerCAM": LayerCAM,
}

# choose the cam method
choosed_cam_methods = "GradCAM" # @param ["GradCAM", "HiResCAM", "ScoreCAM", "GradCAMPlusPlus", "AblationCAM", "XGradCAM", "EigenCAM", "LayerCAM"]

# set model to eval mode and give the last rectified activations maps as reference for computations of the saliency maps.
custom_model.eval()
target_layers = [custom_model[-3]]

# # Instantiate the chosen CAM method
if choosed_cam_methods in cam_methods:
    cam_class = cam_methods[choosed_cam_methods]
    cam = cam_class(model=custom_model, target_layers=target_layers)
else:
    raise ValueError(f"Unsupported CAM method: {choosed_cam_methods}")

# target = None means that the saliency maps will be computed for the highest scoring class of each images.
grayscale_cam = cam(input_tensor=whole_dataset_batch, targets=None)

i = 70 #@param {type: 'integer'}
plt.imshow(np.transpose((whole_dataset_batch[i]/ (1000/225) + 0.5).squeeze().detach().cpu(), (1, 2, 0)), alpha=1)
plt.axis('off')
plt.show()

name_to_save = f"img_{i}.png"

image = np.transpose((whole_dataset_batch[i]/ (1000/225) + 0.5).squeeze().detach().cpu(), (1, 2, 0))
gradCamMaps_tensor = grayscale_cam[i]



# fig,ax = plt.subplots(1,1,figsize=(15,10))

# ax.imshow(gradCamMaps_tensor, cmap = "jet", alpha = 0.7)

# # plot red hatched line
# xline_position = image.shape[0] / 1.32
# ax.axhline(y=xline_position, color='red', linestyle='--', linewidth=4)

# # Add a hatched zone to the left of the vertical line
# ax.fill_betweenx(y=[xline_position-1/2, image.shape[1]-1/2], x1=-1, x2=image.shape[1], color='red', alpha=0.2, hatch='/')

# # fig setup and save
# ax.axis('off')

# # Before saving, adjust the figure's layout
# fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

# # Use 'bbox_inches' and 'pad_inches' to remove the white borders
# fig.savefig(name_to_save, bbox_inches='tight', pad_inches=0)

# plt.show()


#########
# Image #
#########



# # fig, ax = plt.subplots(1, frameon=False)

# img_height, img_width = image.shape[:2]
# # ax.imshow(image, alpha = 1.)

########
# SHAP #
########



# masker_blur = shap.maskers.Image("blur(128,128)", image.shape)

# def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
#     if x.dim() == 4:
#         x = x if x.shape[1] == 3 else x.permute(0, 3, 1, 2)
#     elif x.dim() == 3:
#         x = x if x.shape[0] == 3 else x.permute(2, 0, 1)
#     return x

# def nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
#     if x.dim() == 4:
#         x = x if x.shape[3] == 3 else x.permute(0, 2, 3, 1)
#     elif x.dim() == 3:
#         x = x if x.shape[2] == 3 else x.permute(1, 2, 0)
#     return x

# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]

# transform = [
#     torchvision.transforms.Lambda(nhwc_to_nchw),
#     torchvision.transforms.Lambda(lambda x: x * (1 / 255)),
#     torchvision.transforms.Normalize(mean=mean, std=std),
#     torchvision.transforms.Lambda(nchw_to_nhwc),
# ]

# inv_transform = [
#     torchvision.transforms.Lambda(nhwc_to_nchw),
#     torchvision.transforms.Normalize(
#         mean=(-1 * np.array(mean) / np.array(std)).tolist(),
#         std=(1 / np.array(std)).tolist(),
#     ),
#     torchvision.transforms.Lambda(nchw_to_nhwc),
# ]

# transform = torchvision.transforms.Compose(transform)
# inv_transform = torchvision.transforms.Compose(inv_transform)

# # reverting to images from the shap script
# X, y = shap.datasets.imagenet50()
# Xtr = transform(torch.Tensor(X))

# def predict(img: np.ndarray) -> torch.Tensor:
#     img = nhwc_to_nchw(torch.Tensor(img))
#     img = img.to(device)
#     # choose model here
#     output = custom_model(img)
#     return output

# # Getting ImageNet 1000 class names
# url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
# with open(shap.datasets.cache(url)) as file:
#     class_names = [v[1] for v in json.load(file).values()]
# print("Number of ImageNet classes:", len(class_names))
# # print("Class names:", class_names)

# explainer = shap.Explainer(predict, masker_blur, output_names=class_names)

# topk = 4
# batch_size = 50
# n_evals = 100

# # seems to be wrong format
# # example = transform(image)

# ex_id = 1

# class_id = np.argmax(predict(Xtr[ex_id].unsqueeze(0))[0].detach().numpy())

# shap_values = explainer(
#     Xtr[ex_id:(ex_id+1)],
#     # example.unsqueeze(0),
#     max_evals=n_evals,
#     batch_size=batch_size,
#     outputs=shap.Explanation.argsort.flip[:topk],
# )

# # something goes wrong with the inverse transform
# shap_values.data = inv_transform(shap_values.data).cpu().numpy()[0]
# shap_values.values = [val for val in np.moveaxis(shap_values.values[0], -1, 0)]

# # pembroke is a kind of dog
# shap.image_plot(
#     shap_values=shap_values.values,
#     pixel_values=shap_values.data,
#     labels=shap_values.output_names,
#     # labels=['pembroke', 'candle', 'spotlight', 'digital_clock'],
#     true_labels=[class_names[class_id]],
# )









