import torch

from os import listdir
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


from DASNN import *


def get_args():
    parser = argparse.ArgumentParser(description='Test the network on test images with true labels')

    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')   
    parser.add_argument('--dir-test',  dest="dir_test", type=str, default="../data/datasets/test/", help='Test directory')
    parser.add_argument('--dir-results',type=str, dest="dir_results", default="../results/test/", help='Results directory')
    parser.add_argument('--ntest',  type=int,  default=5, help='Number of tests to do')
     
    
    # Input images arguments
    parser.add_argument('--norm-images', action="store_true", dest="norm_images", default=False, help="Normalize input images")
    parser.add_argument('--amplitude-img', type=float, dest="amplitude_img", default=None, help="Amplitude of images for normalization")
    parser.add_argument('--cbrt', action="store_true", default=False, help="To add if you need to cube root the images before applying the network. Don't use if it's already cube rooted (but don't forget to change amplitude-img).")
    parser.add_argument('--resize-square', default=None, dest="resize_square", help="Method for square resizing (interp or nearest). Put nothing if you don't want resizing")
    
    # Network structure
    parser.add_argument('--shape-mode', default="rect", dest="shape_mode", 
                        help="Convolutional structure of the network. squaresmall, squarebig, rectsame, rectsamemaxpool or rect (default)")
    parser.add_argument('--outFC', default=0, type=int, help="Number of output fully connected hidden layers")
    parser.add_argument('--leaky', action="store_true", default=False, help="For LeakyReLU instead of ReLU")
    parser.add_argument('--batchnorm', action="store_true", default=False, help="To add BatchNorm")

    # Output labels arguments
    parser.add_argument('--labels-out', default="tanh", dest="labels_out", help="Last activation layer can be a tanh (tanh), a relu with (relu_norm) or without (relu) normalizing the values. tanh and relu_norm need to provide max and/or min values")
    parser.add_argument('--path-maxmin-out', dest="path_max_min_out", default=None, help="Path of the txt with the max and min values of the labels for normalization. If nothing is provided, default values are taken")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    device = torch.device('cpu')#'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')
    
    #Initating model
    model = DASNN(shape_mode=args.shape_mode, outfunction=args.labels_out, outFC=args.outFC, leaky=args.leaky, batchnorm=args.batchnorm)
    model = model.to(memory_format=torch.channels_last)
    # Loading model
    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        model.load_state_dict(state_dict)
        print(f'Model loaded from {args.load}')
    print(model)
    model.to(device=device)

    ###
    # Setting predictor
    # All transformations are done into the function predict
    ###

    # Setting input normalization
    if args.norm_images:
        if args.cbrt:
            range_mode="cbrt_norm"
        else:
            range_mode="norm"
    else:
        range_mode="no_range"

    # Setting output normalization
    if args.labels_out == "relu_norm":
        try:
            max_min_outs = np.loadtxt(args.path_max_min_out)
            max_out = max_min_outs[0]
        except:
            print("Max labels : taking default values")
            max_out = np.array([80000, 120, 1])
    elif args.labels_out == "tanh":
        try:
            max_min_outs = np.loadtxt(args.path_max_min_out)
            max_out =  max_min_outs[0]
            min_out = max_min_outs[1]
        except:
            print("Min max labels : taking default values")
            max_out = np.array([80000, 120, 1])
            min_out = np.array([0, 50, 0])
    else: # last case : if labels_out == relu : we do nothing
        max_out = None
        min_out = None


    # Finally, the predictor
    model.eval()
    def predict(image):
        return model.predict(image, og_transform_out=args.labels_out, 
                            range_mode=range_mode, range_input=args.amplitude_img, 
                            max_values=max_out, min_values=min_out,
                            resize_square=args.resize_square)
    


    # Getting test data
    print(args.dir_test)
    labels_dir = listdir(args.dir_test+"labels/")
    labels_dir.sort()
    images_dir = listdir(args.dir_test+"images/")
    images_dir.sort()

    # Test loop
    
    L_true_labs = []
    L_predicted = []
    
    index = np.arange(len(images_dir))
    np.random.shuffle(index)
    for i in range(min(len(images_dir), args.ntest)):
    #for i in index[:args.ntest]:
        
        print("\n",labels_dir[i], images_dir[i])
        
        # Getting data
        label = np.loadtxt(args.dir_test+"labels/"+labels_dir[i])
        image = np.load(args.dir_test+"images/"+images_dir[i])

        L_true_labs.append(label)

        # Prediction
        with torch.no_grad():
            predicted = list(predict(image).numpy().astype(np.float32))
        
        L_predicted.append(predicted)
        
        # Saving result in image
        label_ = list(label.astype(np.float32))
        plt.imshow(image, cmap="bwr", aspect="auto")
        plt.colorbar()
        text = "True : "+str(label_[0])+", "+str(label_[1])+", " + str(int(np.round(label_[2]))) + "\n"
        text += "Predicted : "+str(predicted[0])+", "+str(predicted[1]) + ", "+str(int(np.round(predicted[2]))) + " ("+str(predicted[2]) + ")\n"
        plt.title(text)
        num = ''.join([x for x in images_dir[i] if x.isdigit()])
        plt.savefig(args.dir_results + "res_test_"+num+".png")
        plt.close("all")
        
        # Print results
        print(label_)
        print(predicted)
    
    # Preparing correlation figures
    L_true_labs = np.array(L_true_labs)
    L_predicted= np.array(L_predicted)

    plt.plot(L_true_labs[:,0],L_true_labs[:,0])
    plt.plot(L_true_labs[:,0], L_predicted[:,0], ".")
    plt.xlabel("True weight")
    plt.ylabel("Predicted weight")
    plt.title("Weight")
    plt.savefig(args.dir_results + "correlation_weights.png")
    plt.savefig(args.dir_results + "correlation_weights.pdf")
    plt.close("all")
    
    plt.plot(L_true_labs[:,1],L_true_labs[:,1])
    plt.plot(L_true_labs[:,1], L_predicted[:,1], ".")
    plt.xlabel("True speed")
    plt.ylabel("Predicted speed")
    plt.title("Speed")
    plt.savefig(args.dir_results + "correlation_speed.png")
    plt.savefig(args.dir_results + "correlation_speed.pdf")
    plt.close("all")
    
    plt.plot(L_true_labs[:,2], L_predicted[:,2], ".")
    plt.xlabel("True lane")
    plt.ylabel("Predicted lane")
    plt.title("Lane")
    plt.savefig(args.dir_results + "correlation_lane.png")
    plt.savefig(args.dir_results + "correlation_lane.pdf")
    plt.close("all")

    cm = confusion_matrix(L_true_labs[:,2], np.round(L_predicted[:,2]))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(args.dir_results + "confusion_matrix_lane.png")
    plt.savefig(args.dir_results + "confusion_matrix_lane.pdf")
    plt.close("all")
