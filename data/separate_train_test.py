import numpy as np
import os
from os import listdir, mkdir
from os.path import isdir


np.random.seed(42)

path_imgs = "all_images_cbrt/"
path_labels = "all_labels/"
path_traintest = "datasets_cbrt/"
if not isdir(path_traintest):
    mkdir(path_traintest)
  
path_train=path_traintest + "train/"
if not isdir(path_train):
    mkdir(path_train)
    mkdir(path_train+"images/")
    mkdir(path_train+"labels/")
    
path_test=path_traintest + "test/"
if not isdir(path_test):
    mkdir(path_test)
    mkdir(path_test+"images/")
    mkdir(path_test+"labels/")

list_dir_img = listdir(path_imgs)
list_dir_img.sort()
list_dir_labels = listdir(path_labels)
list_dir_labels.sort()

N = len(list_dir_labels)
M = int(0.9*N)

L_index = np.arange(N)
np.random.shuffle(L_index)


# Training
j=0
for i in L_index[:M]:
    img = np.load(path_imgs + list_dir_img[i])
    labels = np.loadtxt(path_labels + list_dir_labels[i])
    
    print(list_dir_img[i], list_dir_labels[i])
    
    np.save(path_train+"images/img_train_"+str(j)+".npy", img)
    np.savetxt(path_train+"labels/label_train_"+str(j)+".txt", labels)
    
    j+=1 
    
# Test
j=0
for i in L_index[M:]:
    img = np.load(path_imgs + list_dir_img[i])
    labels = np.loadtxt(path_labels + list_dir_labels[i])
    
    np.save(path_test+"images/img_test_"+str(j)+".npy", img)
    np.savetxt(path_test+"labels/label_test_"+str(j)+".txt", labels)
    
    j+=1 
