# DASNN : extracting vehicles characteristics from DAS data

This folder is meant for training and testing a convolutionnal neural network to label characteristics of vechicles with DAS image data with neural networks. 

This README is subdivised in x parts : 
- [Folders description](#folder-description)
- [Requirements](#requirements)
- [Functions and classes description](#functions-and-classes-description)
- [Data details](#data-details)
- [Details about the main script arguments](#details-about-the-main-script-arguments)
- [Details on the convolutional structures](#details-on-the-convolutional-structures)

## Folder description
There are 5 folders : 
- **code** : it contains the code, ie *DASNN.py* where the neural network is implemented and the training script *train.py*
- **data** : it contains the data. Currently, there is is original data before training/test separation, training data, and test data (+ scripts to prepare data). 
- **models** : where the different trained models are saved
- **notebooks** : notebook versions of the training script (empty for the moment)
- **results** : contains the results. For the moment, only contains the subfolder **loss**, with curves about the loss during training

## Requirements
The code needs the following python modules (the versions may not be mandatory, it's only the one I have cuurently for reference)
- **numpy** (1.23.3) : for importing data
- **matplotlib** (3.3.4) : for visualizing loss curves
- **pytorch** (1.10.2) : for initializing training  and using the neural network
- **torchvision** (0.11.3) : for data transformation
- **os** : for importing data
- **argparse** : for parsing arguments when using the main script
- **tqdm** (4.57.0) : for printing training evolution. Not mandatory, there are just a few lines of codes to change if it doesn't work
- **pathlib** : for path gestion. Not mandatory, there are just a few lines of codes to change if it doesn't work.

## Functions and classes description
Let's start with *DASNN.py*. 
### Class DASNN
First, there is the class DASNN, which is the class of the neural network.
It contains the definition of the structures, and the forward function.
There are 5 arguments : the main convolutional structure **shape_mode** (there are for the moment 6 possibilities), the last activation layer **outfunction** (*tanh* or *relu* or *relu*), the number of fully connected layers after the convolutional structure **outFC**, the choice of leakyrelu **leaky** (default is False) instead of a simple ReLU in the layers, and the choice of batch normalization **batchnorm** (default is False).
### Class ImageDAStaset
This class is for dataset management during the training. It defines how to get a new data element (label or image) and using it during training. This is where the optional normalization, and the change of shape (for transforming images into square) is done. It uses the object torchivision.transforms to do theses optional transormations.


Lets' continue with *train.py*
### Function train_model
Function for training the model, with a lot of arguments, such as the model to train, the dataset (with the ImageDAStaset class) to use, the device (CUDA or cpu) to use and a lot of hyper-parameters.

The principal steps of the function are : 
- Splitting the dataset into training and validation
- Creating dataloader (will manage how to access the data by batches)
- Intializing optimizer functions and options
- The training loop

The training loop has different steps : 
- Getting the data
- Putting them on CUDA if asked
- computing the loss
- computing the gradients and taking a step
- printing the step

Each 1/5 of epoch, we do an evaluation round : we plot and save the loss curves (in training and validation data), and if asked, the learning rate is updated if the validation is not bettering.

At each epoch, if asked, the model is saved.

### Main script

The main script has the following steps : 
- It defines the transformations (making images into squares images, normalizations, ... ) that will be done to the data during the training. In particular, if we ask it to normalize the input or output data, it will need to know the maximum and minimum values (see arguments ```amplitude-img``` and ```path-min-max``` of the main script)
- The ImageDAStaset is created with the chosen transforms
- The NN is initialized, enventually loaded from a saved one.
- We launch **train_model.py**


## Data details

Each dataset (training or test) must contain two folders : one for the labels (*label_test_i.txt*) in *txt* format, with all the labels in order separated by a space; the other for the input images (*img_test_i.npy*) in *npy* format, with the same corresponding index. The indices should go from 0 to *dataset length - 1*.

The main data folder contains 4 scripts for data preparation : 
- **stats.py** shows simple statistics of a dataset (to know maxima and minima values for example)
- **visualize.py** helps visualize some of the input images (the effect of transforming them into square images, with different methods...)
- **cbrt.py** makes a new dataset of images from an input one. It will be the same, but cube rooted. It is meant to reveal more small variations in images.
- **separate_train_test.py** separates a full dataset into a training and test set (by default 10\% of the full dataset).

All these scripts have no inputs, you will need to change it directly into the file if needed.

One particular attention point for the dataset folder used for training : it should be composed of two folders : one named *train* and the other *test* (it can be empty for training). Each one should have both labels and images subsets.

To sum up, your folder *datasets* (located in *data*) must have this structure : 
- datasets
  - train
    - images
      - img_train_0.npy
      - img_train_1.npy
      - ...
    - labels
      - label_train_0.txt
      - label_train_1.txt
      - ...
  - test
    - images
      - img_test_0.npy
      - img_test_1.npy
      - ...
    - labels
      - label_test_0.txt
      - label_test_1.txt
      - ...
## Details about the main script arguments

The main script can be simply used, by getting into the **code** folder and entering

```
python3 train.py
```
Some additional arguments can be added.

**Training arguments**
- ```--epochs``` : give the number (```int```) of epochs for the training. Default is 5.
- ```--batch-size``` : give the batch size (```int```). It is better to give a power of 2. Default is 64.
- ```--learning-rate``` : give the (starting) learning rate (```float```). Default is 10e-5.
- ```--autolr``` : to add if you want the learning rate to be adapted throughout the training (based on the score made on the validation set).
- ```--validation``` : give the percent (```float``` or ```int```) of data that will be used for validation. Default is 10.
- ```--print-examples```: to add if you want to print, at each batch, one example of true vs predicted labels.

**Saving arguments**
- ```--load``` : give the path (name included) (```str```) of the pretrained model you want to use if there is one. Default is None.
- ```--save-checkpoints``` : to add if you want to save your model at each step. If not, it will never be saved, even at the end.
- ```--dir-checkpoints``` : give the path (```str```) of the directory where to save models if you used ```--save-checkpoints```. Default is *"../models/"* (ie it will be saved in the models directory)
- ```--name-model``` : give the name (```str```) prefix under which you want to save the models (the suffix will always be *checkpoint_epochi.pth*). Default is ""
- ```--dir-loss``` : give the path (```str```) of the directory where to save loss results. Default is *"../results/loss/"* (ie it will be saved in the *models* directory).

**Dataset argument**
- ```--dir-dataset``` :  give the path (```str```) of the directory of the dataset used for training and test. Default is *"../data/datasets/"* (ie it will be saved in the *models* directory).

**Input images arguments**
- ```--norm-images``` : to add if you want the normalization of the images
- ```--amplitude-img``` : give the range (```float```) of values of the images. For example, if the images go from -0.02 to 0.02, put *0.02* as an argument. Then all values will be transformed from [-0.02; 0.02] to [-1; 1]. Default is None. If you don't put anything, it will take default values (depending on if you used cube rooted images or original images)
- ```--cbrt``` : to add if you want to indicate that you are using cube rooted values for the images. It is only useful if you want to normalize the input images (```norm-images```), and if you don't provide the range (```amplitude-img```); in order to use default values for the cube rooted values. BE CAREFUL : you need to provide the good dataset with already cube rooted images, with ```--dir-dataset```. The main interest of this option is to enhance small variations around 0 for the images with low values, that can help to extract informations.
- ```--resize-square``` : give the method (```str```) to use if we want to resize the images into squares. Only works with the shape-modes (see **Network structure**) with "square". We can just repeat each values 36 times *"nearest"*), or use a bilinear interpolation (*"interp*"). Default is None (no resizing).

**Network structure**
- ```--shape-mode``` : give the main structure (```str```) of the convolutional layers. There are 
6 possible options : *squaresmall*, *squarebig*, *rectsame*, *rectsamemaxpool* or *rect*. Default is *"rect"*.
- ```--outFC``` : give the number (```int```) of fully connected layers after the convolutional layers. Default is 0. For the moment, only 0 and 1 are accepted.
- ```--leaky``` : to add if you want LeakyReLU instead of ReLU as activation layers.
- ```--batchnorm``` : to add if you want to add Batch Normalization after each layer. After a few tests it seems to make the training much better (tested with *labels_out=tanh*).

**Output labels arguments**
- ```--labels-out``` : give the last activation layer (```str```). 3 possible options : *"relu"* (in this case there will be no normalization), *"relu_norm"* (in this case the output training labels will be divided by the max values that can be provided), and *"tanh"* (in this case the output training labels will be normalized between -1 and 1, from the max and min values that can be provided).
- ```--path-maxmin-out``` : give the path + name (```str```) of the file with the max and min values of the labels, for optional normalization. If nothing is put, default values are taken. First line : maxima; second line : minima.
- ```--separate_loss``` : to add if you want to separate the computing of the loss between continous data and binary data. For the moment, only works with [continous, continuous, binary]



To sum up, by default the script will do a training on 5 epochs, with batches of size 64, a learning rate of 1e-5, , without learning rate adaptation, with a validation set of 10\% of the total set, from a blank NN, without saving the NNs, with normalization of the images that are not cube rooted and not resized to squares, with the *"rect"* structure, with no additional fully connected layer after convolution, no batchnorm, only ReLU activation functions, working with labels between -1 and 1 (computed from default values).

For the moment, after a few tests, here is a good configuration I'd recommend for starting :  
```
python3 train.py --norm-images --labels-out tanh --autolr --batchnorm
```

## Details on the convolutional structures
There are two main possible convolutional structures : square and rectangle. In the square one, we transform the images into squares (by repeating valyes or interpolating). Therefore the convolutional structures are meant to work with squared 900x900 images. In the rectangle one, we let the images as rectangles. The convolutional structures are meant to work with 900x25 images.

The square structure will require much more time and GPU.
The rectangle structures will require much less time and GPU but I'm not sure about myself if the chosen structure are very effective.

If, after testing all the possibilities, we see that the square structures don't show particularly better prediction properties than the rectangle structures, then there is no point in using square structure !

More details with the structures : 

- *"squaresmall"* : There are few (5) convolutional layers with a big kernel. At each step the dimension decreases a lot.
- *"squarebig"* : More steps, smaller kernel. The dimension decreases slower.
- *"rectsame"* : Uses square kernels on rectangle images. Therefore, each dimension is treated the same way despite the asymetry of the images. Maybe we overuse vertical information compared to horizontal information here. It finishes with a 56x1 image before the fully connected layers.
- *"rectsamemaxpool"* : Same thing than "rectsame", except after each convolution, a maxpool layer is applied only to the vertical dimension, ie we divide the vertical dimension two times more than the horizontal one. It limits the asymetry.
- *"rect"* : Asymetrical convolutions with rectangular kernels and strides. The first convolution is only done vertically
- *"rectplus"* : Same thing, but with vertical convolutions from the beginning and with one more step.

Helpful websites to design convolutional structures : 
- [layer-calc.com](http://layer-calc.com)
- [madebyollin.github.io/convnet-calculator](madebyollin.github.io/convnet-calculator)