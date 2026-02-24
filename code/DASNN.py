import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.utils.data import Dataset
import os
from os import listdir
import numpy as np
flattening = nn.Flatten()
unflattening = nn.Unflatten(1,(900,25))

import matplotlib.pyplot as plt


class DASNN(torch.nn.Module):
    def __init__(self, shape_mode="squaresmall", outfunction="tanh", outFC=0, leaky=False, batchnorm=False,inFC=False,dropout=False):
        super().__init__()
        self.shape_mode = shape_mode
        self.outfunction=outfunction
        self.outFC=outFC
        self.inFC=inFC
        self.leaky=leaky
        self.batchnorm=batchnorm
        self.maxpool=False
        self.dropout=dropout

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
        
        
        if "rect" in self.shape_mode and self.inFC:
            self.inFC1 = nn.Linear(900*25, 900*25)
        

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
            if self.dropout:
                layers.append(nn.Dropout(p=0.2))
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
        if self.outFC == 1:
            intermediate = (self.outconvsize*self.nb_channels*2**(self.nb_steps) + self.outsize)//2
            self.FC1 = nn.Linear(self.outconvsize*self.nb_channels*2**(self.nb_steps), intermediate)
            #if self.batchnorm:
            #    self.FC1 = nn.Sequential(self.FC1, nn.BatchNorm1d(num_features=intermediate))
            self.intermediatefunc = nn.ReLU(True)
            self.FC2 = nn.Linear(intermediate, self.outsize)
            #if self.batchnorm:
            #    self.FC2 = nn.Sequential(self.FC2, nn.BatchNorm1d(num_features=self.outsize))
        elif self.outFC > 0:

            intermediate = (self.outconvsize*self.nb_channels*2**(self.nb_steps) + self.outsize)//2
            self.FC1 = nn.Linear(self.outconvsize*self.nb_channels*2**(self.nb_steps), intermediate)
            #if self.batchnorm:
            #    self.FC1 = nn.Sequential(self.FC1, nn.BatchNorm1d(num_features=intermediate))
            self.intermediatefunc = nn.ReLU(True)
            
            self.FC2 = nn.Linear(intermediate, intermediate)
            self.FC3 = nn.Linear(intermediate, self.outsize)
        else:
            self.FC = nn.Linear(self.outconvsize*self.nb_channels*2**(self.nb_steps), self.outsize)
            #if self.batchnorm:
            #    self.FC = nn.Sequential(self.FC, nn.BatchNorm1d(num_features=self.outsize))

        
        if self.outfunction == "tanh":
            self.outlayer = nn.Tanh()
        else:
            self.outlayer = nn.ReLU(True)

    def forward(self, x):
    
        if "rect" in self.shape_mode and self.inFC:
            x = flattening(x)
            x = self.inFC1(x)
            x = unflattening(x)
            
        out_conv = self.main_module(x)
        out_conv = flattening(out_conv)
        if self.outFC == 1:
            outFC = self.FC2(self.intermediatefunc(self.FC1(out_conv)))
        elif self.outFC > 0:
            outFC = self.FC3(self.intermediatefunc(self.FC2(self.intermediatefunc(self.FC1(out_conv)))))
        else:
            outFC = self.FC(out_conv)
        return self.outlayer(outFC)
    
    def predict(self, image, og_transform_out="tanh", range_mode="norm", range_input=None, 
                      max_values=None, min_values=None,resize_square=None):
        
        image_ = image.copy()
        print("Range mode : ",range_mode)

        if "cbrt" in range_mode:
            image_ = np.cbrt(image_)
        if "norm" in range_mode:
            if range_input is None:
                if range_mode == "norm":
                    range_input = 0.00015  # 0.0005563842691301197 with noise
                else: #elif range_mode == "cbrt_norm"
                    range_input = 0.06  # Noisy images : 0.08188320695692161
            image_ = image_ / range_input
            
        image_ = torch.Tensor(image_)

        image_batch = torch.unsqueeze(torch.unsqueeze(image_, 0), 0)
        
        if resize_square is not None:
            if resize_square == "interp":
                image_batch = nnf.interpolate(image_batch, size=(900,900), mode="bilinear")
            else:# elif resize_square == "nearest":
                image_batch = nnf.interpolate(image_batch, size=(900,900), mode="nearest")

        results = self.forward(image_batch)[0].detach().cpu()
        
        if og_transform_out == "tanh":
            if max_values is None or min_values is None:
                max_values = np.array([80000, 120, 1])
                min_values = np.array([0, 50, 0])
            mu_values = (max_values + min_values)/2
            sig_values = (max_values - min_values)/2
            results = results*sig_values + mu_values
        elif og_transform_out == "relu_norm":
            if max_values is None:
                mu_values = np.array([80000, 120, 1])
                sig_values = np.array([0, 0, 0])
            else:
                mu_values = np.zeros(len(max_values))
                sig_values = max_values
            results = results*sig_values + mu_values

        #print(max_values, min_values, sig_values, mu_values)
        #print(results)
        return results

class ImageDAStaset(Dataset):
    def __init__(self, root, img_transform=None, label_transform=None, mode_set="train", mu_labels=None, sig_labels=None,with_noise=False,progressive_noise=False,changing_noise=False):
        self.root_dir=root
        self.img_dir=root+mode_set+"/images/"
        self.label_dir=root+mode_set+"/labels/"
        self.img_noisy_dir=root+mode_set+"/noisy/"
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.mode_set = mode_set
        self.with_noise = with_noise
        self.progressive_noise = progressive_noise
        self.changing_noise = changing_noise
        self.list_dir_noise = os.listdir("../data/all_noise")
        
        if mu_labels is None or sig_labels is None:
            self.transform_values = False
        else:
            self.transform_values = True
            self.mu_labels = mu_labels
            self.sig_labels = sig_labels

    def __len__(self):
        return len(listdir(self.img_dir))
    def __getitem__(self, idx):
        if self.with_noise:
            img_path = self.img_noisy_dir+"img_"+self.mode_set+"_"+str(idx)+".npy"
        else:
            img_path = self.img_dir+"img_"+self.mode_set+"_"+str(idx)+".npy"
           
                
                
                
        if self.progressive_noise:
            img_noise_path = self.img_noisy_dir+"img_"+self.mode_set+"_"+str(idx)+".npy"
            image_noise = np.load(img_noise_path)
        
        image = np.load(img_path)
        
        if self.changing_noise:
            noise = np.load(f"allnoise/{np.random.choice(list_dir_noise)}")
            range_noise = np.max(noise) - np.min(noise)
            if "cbrt" in self.root_dir:
                image = image**3
            range_img = np.max(img) - np.min(img)
            fact = np.random.rand()*0.8+0.2
            noise *= (range_img/range_noise)/fact
            image = image+noise
            if "cbrt" in self.root_dir:
                image = np.cbrt(image)
        
        label_path = self.label_dir+"label_"+self.mode_set+"_"+str(idx)+".txt"
        if self.transform_values:
            label = (torch.Tensor(np.loadtxt(label_path)) - self.mu_labels )/ self.sig_labels
        else:
            label = torch.Tensor(np.loadtxt(label_path))

        if self.img_transform:
            image = self.img_transform(image)
            if self.progressive_noise:
                image_noise = self.img_transform(image_noise)
        if self.label_transform:
            label = self.label_transform(label)
        if self.progressive_noise:
            return image, label, image_noise
        else:
            return image, label
