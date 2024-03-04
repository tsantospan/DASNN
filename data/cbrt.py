import numpy as np
import os
from os import listdir

path_imgs = "all_images/"
path_cbrt = "all_images_cbrt/"
list_dir_img = listdir(path_imgs)

for name in list_dir_img:
    img = np.load(path_imgs + name)
    img = np.cbrt(img)
    np.save(path_cbrt+name, img)
