# pylint: disable=missing-function-docstring
import argparse
import os
import time


import torch
#from sklearn.metrics import roc_curve
#from sklearn.metrics import roc_auc_score
from utils.deformation import prob_heatmap_tensor,warped_imgs
from utils.classifier import initialize_patch_model
from utils.train_util import initialize_data_loader, initialize_whole_model,first_stage,second_stage
from data.split import split_dataset




def parse_args():
    parser = argparse.ArgumentParser(description='train, resume, test arguments')
    parser.add_argument('--project_root', default= os.getcwd(), type = str)
    parser.add_argument('--data_localization',default ='/home/sposso22/work/final_data/loc.csv',type=str, help= 'data paths')
    parser.add_argument('--aug', default = False, action="store_true")
    parser.add_argument('--batch_size', '-b',default=10, type = int, help = "mini-batch size per worker(GPU)" )
    parser.add_argument('--workers', '-w', default= 8, type = int, help ="Number of data loading workers")
    parser.add_argument('--warp', default = True, action="store_true")
    parser.add_argument('--res', '-r', default=1, type= int, help='choose wanted resolution from list')
    parser.add_argument('--sigma', '-S', default= 14, type = int, help ="Sigma value of the Gaussian Kernel")
    #parser.add_argument("--checkpoint-file",default=os.getcwd()+"/tmp/checkpoint.pth.tar",type=str,help="checkpoint file path, to load and save to")
    parser.add_argument('--epochs','-e', default = 50, type = int, help='number of total epochs to run')


    return parser.parse_args()



def train_model(model, dataloaders, criterion,optimizer,patch_classifier,warp,res, sigma):
    
    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for img_tensor,labels in dataloaders[phase]:
    
            if warp:
                heatmaps = prob_heatmap_tensor(img_tensor,patch_classifier)
                sampled_imgs = warped_imgs(img_tensor,heatmaps,res,sigma)

                inputs= sampled_imgs.expand(-1,3,*sampled_imgs.shape[2:])
                #inputs = inputs.to(device = device, dtype = torch.float)
                
            else :
                inputs=img_tensor.expand(-1,3,*img_tensor.shape[2:])
                #inputs = inputs.to(device = device, dtype = torch.float)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                # Get model outputs and calculate loss
                # Special case for inception because in training it has an auxiliary output. In train
                #   mode we calculate the loss by summing the final output and the auxiliary output
                #   but in testing we only consider the final output


                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

        print('Phase: {} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        
        if phase == 'val':
            
            val_acc = epoch_acc


        print()
    
    return val_acc



def main():

   
    args = parse_args() 
    res = [(288,224),(576,448),(864,672),(1152,896)][args.res]
    
    #Initilialize data  
    split_dataset(args.data_localization)

    #Initilalize patch classifier
    patch_classifier = initialize_patch_model("resnet", num_classes=5,use_pretrained = True, root = args.project_root, useLightning=True)
    
    # Initilaize whole image classifier and loss_function (CrossEntropyLoss)
    model, criterion = initialize_whole_model(args.project_root)

    #The training process consists of two stages, so we set different optimizer for each stage.
    
    first_optimizer = first_stage(model)
    second_optimizer = second_stage(model)

    train_loader,val_loader, test_loader = initialize_data_loader(args.batch_size,args.workers,args.project_root,args.aug)

    dataloaders_dict = {'train':train_loader,'val': val_loader}


    #state = load_checkpoint(args.checkpoint_file, device_id, model, second_optimizer)

    

    since = time.time()


    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)

        #state.epoch = epoch
        #train_loader.batch_sampler.sampler.set_epoch(epoch)

        if epoch <30:

            acc = train_model(model, dataloaders_dict, criterion,first_optimizer,patch_classifier,args.warp,res,args.sigma)

        else:

            acc = train_model(model, dataloaders_dict, criterion,second_optimizer,patch_classifier,args.warp,args.res,args.sigma)
        
        #is_best = acc>state.best_acc1
        #state.best_acc1 = max(acc,state.best_acc1)
        
        #if device_id ==0:
            #save_checkpoint(state, is_best, args.checkpoint_file)
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:4f}'.format(state.best_acc1))



if __name__ == '__main__':
    main()
