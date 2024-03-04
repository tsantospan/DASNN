import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
from os import listdir
import numpy as np
flattening = nn.Flatten()
class DASNN(torch.nn.Module):
    def __init__(self, shape_mode="squaresmall", outfunction="tanh", outFC=0, leaky=False, batchnorm=False):
        super().__init__()
        self.shape_mode = shape_mode
        self.outfunction=outfunction
        self.outFC=outFC
        self.leaky=leaky
        self.batchnorm=batchnorm
        self.maxpool=False

        self.outsize=3
        

        # Choice of the structure and corresponding hyper parameters
        if self.shape_mode == "squaresmall":
            # For a 900x900 square
            self.kernels = 5
            self.strides = 3
            self.outconvsize = 9
            self.nb_steps = 5
            self.nb_channels = 32

        elif self.shape_mode == "squarebig":
            # For a 900x900 square; more convolutions
            self.kernels = 4
            self.strides = 2
            self.outconvsize = 49
            self.nb_steps = 7
            self.nb_channels = 16

        elif self.shape_mode == "rectsame":
            self.kernels = 4
            self.strides = 2
            self.outconvsize = 56
            self.nb_steps = 4
            self.nb_channels = 64

        elif self.shape_mode == "rectsamemaxpool":
            self.kernels = 4
            self.strides = 2
            self.outconvsize = 7
            self.nb_steps = 4
            self.nb_channels = 64
            self.maxpool=True

        elif self.shape_mode == "rectplus":
            self.kernels = (10,3)
            self.strides = (4,2)
            self.outconvsize = 1
            self.nb_steps = 5
            self.nb_channels = 64

        else: #elif self.shape_mode == "rect":
            self.kernels = (10,3)
            self.strides = (4,2)
            self.outconvsize = 8
            self.nb_steps = 4
            self.nb_channels = 64
        layers = []
        

        # Definition fo the different convolutional layers
        for i in range(self.nb_steps):
            if i == 0:
                # Specific case : we start with one channel
                if shape_mode == "rect":
                    layers.append(nn.Conv2d(
                        in_channels=1,
                        out_channels=self.nb_channels*2**(i+1),
                        kernel_size=self.kernels,
                        stride=(4,1), # Specific case to not decrease the horizontal dimension at the first step
                        padding=1
                        )
                        )
                else:
                    layers.append(nn.Conv2d(
                        in_channels=1,
                        out_channels=self.nb_channels*2**(i+1),
                        kernel_size=self.kernels,
                        stride=self.strides,
                        padding=1
                        )
                        )
            elif i == self.nb_steps -1 and shape_mode=="rectplus":
                layers.append(nn.Conv2d(
                    in_channels=self.nb_channels*2**i,
                    out_channels=self.nb_channels*2**(i+1),
                    kernel_size=2, # Specific case : at this point we have a 2x2 image
                    stride=1,
                    padding=0
                    )
                    )
            else:
                layers.append(nn.Conv2d(
                    in_channels=self.nb_channels*2**i,
                    out_channels=self.nb_channels*2**(i+1),
                    kernel_size=self.kernels,
                    stride=self.strides,
                    padding=1
                    )
                    )
            if self.batchnorm:
                layers.append(nn.BatchNorm2d(num_features=self.nb_channels*2**(i+1)))
            if self.leaky:
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            else:
                layers.append(nn.ReLU(True))
            if self.maxpool and i < self.nb_steps-1:
                layers.append(nn.MaxPool2d((2,1)))


        self.main_module = nn.Sequential(*layers)
        # Summary of the convolutional structure: 
        # Alternance between :
        #  - Conv2d (rectangular or square) (the first one is different for rect)
        #  - Batchnorm if asked
        #  - ReLU or LeakyReLU
        #  - MaxPool for rectsamemaxpool (except last layer)

        # Now let's go with the final layers
        if self.outFC > 0:
            intermediate = (self.outconvsize*self.nb_channels*2**(self.nb_steps) + self.outsize)//2
            self.FC1 = nn.Linear(self.outconvsize*self.nb_channels*2**(self.nb_steps), intermediate)
            #if self.batchnorm:
            #    self.FC1 = nn.Sequential(self.FC1, nn.BatchNorm1d(num_features=intermediate))
            self.intermediatefunc = nn.ReLU(True)
            self.FC2 = nn.Linear(intermediate, self.outsize)
            #if self.batchnorm:
            #    self.FC2 = nn.Sequential(self.FC2, nn.BatchNorm1d(num_features=self.outsize))
        else:
            self.FC = nn.Linear(self.outconvsize*self.nb_channels*2**(self.nb_steps), self.outsize)
            #if self.batchnorm:
            #    self.FC = nn.Sequential(self.FC, nn.BatchNorm1d(num_features=self.outsize))

        
        if self.outfunction == "tanh":
            self.outlayer = nn.Tanh()
        else:
            self.outlayer = nn.ReLU(True)

    def forward(self, x):
        out_conv = self.main_module(x)
        out_conv = flattening(out_conv)
        if self.outFC > 0:
            outFC = self.FC2(self.intermediatefunc(self.FC1(out_conv)))
        else:
            outFC = self.FC(out_conv)
        return self.outlayer(outFC)

class ImageDAStaset(Dataset):
    def __init__(self, root, img_transform=None, label_transform=None, mode_set="train", mu_labels=None, sig_labels=None):
        self.root_dir=root
        self.img_dir=root+mode_set+"/images/"
        self.label_dir=root+mode_set+"/labels/"
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.mode_set = mode_set
        
        if mu_labels is None or sig_labels is None:
            self.transform_values = False
        else:
            self.transform_values = True
            self.mu_labels = mu_labels
            self.sig_labels = sig_labels

    def __len__(self):
        return len(listdir(self.img_dir))
    def __getitem__(self, idx):
        img_path = self.img_dir+"img_"+self.mode_set+"_"+str(idx)+".npy"
        image = np.load(img_path)
        label_path = self.label_dir+"label_"+self.mode_set+"_"+str(idx)+".txt"
        if self.transform_values:
            label = (torch.Tensor(np.loadtxt(label_path)) - self.mu_labels )/ self.sig_labels
        else:
            label = torch.Tensor(np.loadtxt(label_path))

        if self.img_transform:
            image = self.img_transform(image)
        if self.label_transform:
            label = self.label_transform(label)

        return image, label