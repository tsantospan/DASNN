import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch import optim
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from os import mkdir
from os.path import isdir

from DASNN import *

def train_model(
        model,
        device,
        dataset,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        auto_lr: bool = True,
        dir_checkpoint: str = "../models/",
        name_model: str = "",
        dir_loss: str = "../results/loss/",
        separate_loss: bool = False,
        labels_out: str = "tanh",
        print_example: bool = False
):
    
    # 1. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])#, generator=torch.Generator().manual_seed(0))

    # 2. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    num_val_batches = len(val_loader)


    print(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
    ''')

    # 3. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = optim.RMSprop(model.parameters(),
    #                          lr=learning_rate, weight_decay=1e-8, momentum=0.999)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
    criterion = nn.MSELoss()
    if separate_loss:
        if labels_out == "tanh":
            #criterion_class = nn.HingeEmbeddingLoss()
            criterion_class = nn.BCELoss()
        else:
            criterion_class = nn.BCELoss()

    global_step = 0
    
    # For loss following
    L_loss_training = []
    L_loss_valid = []
    L_lr = []
    
    # 4. Begin training
    for epoch in range(1, epochs + 1):
        model.train() # Training mode
        
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_labels = batch[0], batch[1]
                #print(torch.max(images[0]))
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_labels = true_labels.to(device=device, dtype=torch.float32)
 
                labels_pred = model(images)
                if print_example:
                    print("\nTrue labels (normed)     :",true_labels[0],"\nPredicted labels (normed):",labels_pred[0])
                if separate_loss:
                    loss = criterion(labels_pred[:,:2], true_labels[:,:2])
                    if labels_out == "tanh":
                        loss += criterion_class((labels_pred[:,[2]]+1)/2, (true_labels[:,[2]]+1)/2)#/batch_size
                    else:
                        loss += criterion_class(labels_pred[:,[2]], (true_labels[:,[2]]+1)/2)/batch_size
                else:
                    loss = criterion(labels_pred, true_labels)
                    
                L_loss_training.append(float(loss.detach().cpu().numpy()))
                L_lr.append(optimizer.param_groups[0]['lr'])

                optimizer.zero_grad(set_to_none=True)
                activate_grad_scaler=False
                if not activate_grad_scaler:
                    loss.backward()
                    optimizer.step()
                else:
                    grad_scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

                pbar.update(images.shape[0])
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                global_step += 1

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    # Saving plot of losses
                    if global_step % division_step == 0:
                        if not isdir(dir_loss):
                            mkdir(dir_loss)
                        plt.figure()
                        plt.plot(L_loss_training)
                        plt.savefig(dir_loss+"training.png")
                        plt.close("all")
                        plt.figure()
                        plt.semilogy(L_loss_training)
                        plt.savefig(dir_loss+"training_log.png")
                        plt.close("all")
                        plt.figure()
                        plt.plot(L_lr)
                        plt.savefig(dir_loss+"lr.png")
                        plt.close("all")
                        plt.figure()
                        plt.semilogy(L_lr)
                        plt.savefig(dir_loss+"lr_log.png")
                        plt.close("all")

                        # Evaluation
                        model.eval() # Evaluation mode
                        score = 0

                        with torch.no_grad():
                            for batch in tqdm(val_loader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
                                image, label_true = batch[0], batch[1]

                                # move images and labels to correct device and type
                                image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                                label_true = label_true.to(device=device, dtype=torch.float32)
            
                                # predict
                                pred_label = model(image)

                                if print_example:
                                    print("\nValidation :\nTrue labels (normed)     :",label_true[0],"\nPredicted labels (normed):",pred_label[0])
                                
                                if not isdir(dir_loss+"validation_results"):
                                    mkdir(dir_loss+"validation_results")
                                plt.plot(label_true[:,0].cpu(), label_true[:,0].cpu())
                                plt.plot(label_true[:,0].cpu(), pred_label[:,0].cpu(), ".")
                                plt.xlabel("True weight")
                                plt.ylabel("Predicted weight")
                                plt.title("Weight")
                                plt.savefig(dir_loss+"validation_results/"+"weight_step"+str(global_step)+".png")
                                plt.close("all")
                                plt.plot(label_true[:,1].cpu(), label_true[:,1].cpu())
                                plt.plot(label_true[:,1].cpu(), pred_label[:,1].cpu(), ".")
                                plt.xlabel("True speed")
                                plt.ylabel("Predicted speed")
                                plt.title("Speed")
                                plt.savefig(dir_loss+"validation_results/"+"speed_step"+str(global_step)+".png")
                                plt.close("all")
                                plt.plot(label_true[:,2].cpu(), label_true[:,2].cpu())
                                plt.plot(label_true[:,2].cpu(), pred_label[:,2].cpu(), ".")
                                plt.xlabel("True lane")
                                plt.ylabel("Predicted lane")
                                plt.title("Lane")
                                plt.savefig(dir_loss+"validation_results/"+"lane_step"+str(global_step)+".png")
                                plt.close("all")

                            if separate_loss:
                                score += criterion(pred_label[:,:2], label_true[:,:2])
                                if labels_out == "tanh":
                                    loss += criterion_class((labels_pred[:,[2]]+1)/2, (true_labels[:,[2]]+1)/2)#/batch_size
                                else:
                                    loss += criterion_class(labels_pred[:,[2]], (true_labels[:,[2]]+1)/2)/batch_size

                                print("MSE loss :", criterion(pred_label[:,:2], label_true[:,:2]))

                                if labels_out == "tanh":
                                    print("Class loss :", criterion_class((labels_pred[:,[2]]+1)/2, true_labels[:,[2]]))#/batch_size)
                                else:
                                    print("Class loss :",criterion_class(labels_pred[:,[2]], true_labels[:,[2]])/batch_size)
                            else:
                                score += criterion(pred_label, label_true)

                        model.train() # Back to training mode
                        val_score = score / max(num_val_batches, 1)
                        
                        if auto_lr:
                            scheduler.step(val_score)
                        
                        # Saving plot of losses
                        L_loss_valid.append(float(val_score.cpu().numpy()))
                        plt.figure()
                        plt.plot(L_loss_valid)
                        plt.savefig(dir_loss+"valid.png")
                        plt.close("all")
                        plt.figure()
                        plt.semilogy(L_loss_valid)
                        plt.savefig(dir_loss+"valid_log.png")
                        plt.close("all")

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, str(Path(dir_checkpoint) / (name_model+'checkpoint_epoch{}.pth'.format(epoch))))
            print(f'Checkpoint {epoch} saved!')




def get_args():
    parser = argparse.ArgumentParser(description='Train the network on images and target labels')
    # Training arguments
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=64, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='(Initial) learning rate', dest='lr')
    parser.add_argument('--autolr', action="store_true", default=False, help="For changing lr value automatically") 
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--print-examples', action="store_true", default=False, dest="print_example",
                        help="If you want to show one example of true vs predicted labels at each batch")

    # Saving arguments
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--save-checkpoints', action="store_true", dest="save_checkpoint", default=False, 
                        help="To save checkpoints of the model at each epoch")    
    parser.add_argument('--dir-checkpoints', dest="dir_checkpoint", default="../models/", 
                        help="Directory to save models")    
    parser.add_argument('--name-model', dest="name_model", default="", help="Optional prefix name model")    
    parser.add_argument('--dir-loss', dest="dir_loss", default="../results/loss/", help="Directory to save loss")    
    
    # Dataset argument
    parser.add_argument('--dir-dataset', dest="dir_dataset", default="../data/datasets/", help="Dataset directory")    

    # Input images arguments
    parser.add_argument('--norm-images', action="store_true", dest="norm_images", default=False, help="Normalize input images")
    parser.add_argument('--amplitude-img', type=float, dest="amplitude_img", default=None, help="Amplitude of images for normalization")
    parser.add_argument('--cbrt', action="store_true", default=False, help="Indicates if the images inputs have been cube rooted; for normalization")
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
    parser.add_argument('--separate-loss', action="store_true", default=False, dest="separate_loss", help="For separating the loss between continuous and class outputs")
    
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')
 

    #######################
    # Dataset preparation #
    #######################

    # Transforms of the images
    img_transforms = [transforms.ToTensor()]
    if args.norm_images:
        if args.amplitude_img is None:
            if args.cbrt:
                print("Min max images : taking default cbrt values.")
                mini_img = -0.06
                maxi_img = 0.06
                if args.dir_dataset == "../data/datasets/": # Just in case someone forgets to provide the appropriate dataset directory
                    args.dir_dataset = "../data/datasets_cbrt/"
            else:
                print("Min max images : taking default values.")
                mini_img = -0.00014
                maxi_img = 0.00014
        else:
            mini_img = - args.amplitude_img
            maxi_img = args.amplitude_img

        mu_img = (maxi_img + mini_img)/2
        sig_img = (maxi_img - mini_img)/2

        img_transforms.append(transforms.Normalize((mu_img,), (sig_img,)))


    if "square" in args.shape_mode:
        if args.resize_square == "interp":
            img_transforms.append(transforms.Resize((900,900), interpolation=InterpolationMode.BILINEAR))
        else:
            img_transforms.append(transforms.Resize((900,900), interpolation=InterpolationMode.NEAREST))

        
    # Transforms of the labels
    label_transforms = [
            #transforms.ToTensor(),
            transforms.functional.convert_image_dtype
            ]

    if args.labels_out == "relu_norm":
        try:
            max_min_outs = np.loadtxt(args.path_max_min_out)
            max_out = max_min_outs[0]
        except:
            print("Max labels : taking default values")
            max_out = [80000, 120, 1]
        
        mu_labels = torch.zeros(len(max_out))
        sig_labels = torch.Tensor(max_out)
        
    elif args.labels_out == "tanh":
        try:
            max_min_outs = np.loadtxt(args.path_max_min_out)
            max_out =  max_min_outs[0]
            min_out = max_min_outs[1]
        except:
            print("Min max labels : taking default values")
            max_out = np.array([80000, 120, 1])
            min_out = np.array([0, 50, 0])

        mu_labels = torch.Tensor((max_out + min_out)/2)
        sig_labels = torch.Tensor((max_out - min_out)/2)

    else: # last case : if labels_out == relu : we do nothing
        mu_labels = None
        sig_labels = None
    
    dataset = ImageDAStaset(root = args.dir_dataset,
        img_transform = transforms.Compose(img_transforms),
        label_transform=transforms.Compose(label_transforms),
        mu_labels=mu_labels,
        sig_labels=sig_labels
        )

    ########################
    # Model initialization #
    ######################## 
    model = DASNN(shape_mode=args.shape_mode, outfunction=args.labels_out, outFC=args.outFC, leaky=args.leaky, batchnorm=args.batchnorm)
    model = model.to(memory_format=torch.channels_last)

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        model.load_state_dict(state_dict)
        print(f'Model loaded from {args.load}')
    print(model)
    model.to(device=device)
    

    ############
    # Training #
    ############
    train_model(
            model=model,
            dataset=dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            val_percent=args.val / 100,
            save_checkpoint=args.save_checkpoint,
            auto_lr=args.autolr,
            dir_checkpoint=args.dir_checkpoint,
            name_model=args.name_model,
            dir_loss=args.dir_loss,
            separate_loss=args.separate_loss,
            labels_out=args.labels_out,
            print_example=args.print_example
        )

