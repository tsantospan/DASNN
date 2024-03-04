import numpy as np
import matplotlib.pyplot as plt
import os
from os import listdir

list_dir = os.listdir("all_labels")

L_labels = []


print("Statistics about labels...")
for name in list_dir:
    L_labels.append(np.loadtxt("all_labels/"+name))

L_labels = np.array(L_labels)

plt.figure()
plt.hist(L_labels[:,0])
plt.title("Weight")

plt.figure()
plt.hist(L_labels[:,1])
plt.title("Speed")

plt.figure()
plt.hist(L_labels[:,2])
plt.title("Lane")

plt.show()


L_maxs = []
L_mins = []
L_stds = []
L_meanabs = []

print("Statistics about images...")
list_dir = os.listdir("all_images_sqrt3")
for name in list_dir:
    A = np.load("all_images_sqrt3/"+name)
    L_maxs.append(np.max(A))
    L_mins.append(np.min(A))
    L_stds.append(np.std(A))
    L_meanabs.append(np.mean(abs(A)))
    
plt.figure()
plt.hist(L_mins)
plt.title("Mins")


plt.figure()
plt.hist(L_maxs)
plt.title("Maxs")

plt.figure()
plt.hist(L_stds)
plt.title("Std")

plt.figure()
plt.hist(L_meanabs)
plt.title("Mean abs")



plt.figure()
plt.hist(np.log(L_maxs))
plt.title("Maxs")

plt.figure()
plt.hist(np.log(L_stds))
plt.title("Std log")

plt.figure()
plt.hist(np.log(L_meanabs))
plt.title("Mean abs")

plt.show()
    
