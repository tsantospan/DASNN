import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as nnf
import os
from os import listdir

list_dir = listdir("all_images")

mode="autovminmax"
# Modes : "vminmax", "cbrt", "autovminmax"
if mode=="cbrt":
    vmini = np.cbrt(-0.00014)
    vmaxi = np.cbrt(0.00014)
elif mode=="vminmax":
    vmini = -0.00014
    vmaxi = 0.00014
else:
    vmini=None
    vmaxi=None
    
for name in list_dir:
    if mode=="cbrt":
        A_ = np.cbrt(np.load("all_images/"+name))
    else:
        A_ = np.load("all_images/"+name)
    A = A_.copy()

    for i in range(36):
        A = np.concatenate((A,A_), axis=1)
    plt.figure()
    plt.imshow(A, cmap="bwr", vmin=vmini, vmax=vmaxi)
    plt.colorbar()

    A2 = nnf.interpolate(torch.Tensor([[A_]]), size=(900,900), mode="bilinear")
    plt.figure()
    plt.imshow(A2[0,0], cmap="bwr",vmin=vmini, vmax=vmaxi)
    plt.colorbar()

    A3 = nnf.interpolate(torch.Tensor([[A_]]), size=(900,900), mode="bicubic")
    plt.figure()
    plt.imshow(A3[0,0], cmap="bwr",vmin=vmini, vmax=vmaxi)
    plt.colorbar()

    A4 = np.repeat(A_, 36, axis=1)
    plt.figure()
    plt.imshow(A4, cmap="bwr",vmin=vmini, vmax=vmaxi)
    plt.colorbar()


    plt.show()
    
    
