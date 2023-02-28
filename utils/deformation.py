import numpy as np
import torch
from torch import nn

import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import  DataLoader


def prob_heatmap_tensor(img_tensor, patch_classifier,mean_=0.2939,std_=0.2694,
                        batch_size=16,patch_size=224, stride=32, padding = 111):
    '''Sweep image data with a trained model to produce prob heatmaps
    '''
    nb_row = round(float(img_tensor.shape[2])/stride )
    nb_col = round(float(img_tensor.shape[3])/stride )
    nb_row = int(nb_row)
    nb_col = int(nb_col)
   
    heatmap_list = []
    unfold = torch.nn.Unfold(kernel_size=(patch_size,patch_size), dilation=1, padding= padding, stride=stride)
    for img in img_tensor:
        img = img.unsqueeze(dim=0)
        img_= unfold(img)
        img_ = img_.permute(0,2,1)
        n = img_.size(0)
        c = img_.size(1)
        img_ = img_.view(c,1,224,224)
        patch_X = img_.expand(-1,3,-1,-1)

        patch_classifier.eval()
        with torch.no_grad():
            patch_X = patch_X.clone().detach().requires_grad_(False)
            transform = T.Normalize(mean=[mean_,mean_,mean_],std=[std_,std_,std_])
            patch_X = transform(patch_X.clone())
            loader = DataLoader(patch_X, batch_size)
            preds =[]
            
            for i in loader:
                #i = i #.to(device)
                chunk_preds = patch_classifier(i)
                chunk_preds = nn.functional.softmax(chunk_preds, dim=1)  
                preds.append(chunk_preds)

          
            pred = torch.vstack(preds)
        
            pred = pred[:,1:5].sum(axis=1)
        

            heatmap = pred.reshape((nb_row, nb_col))
            
            heatmap_list.append(heatmap.unsqueeze(dim=0))
        
        heatmap_tensor = torch.cat(heatmap_list)
            
    return heatmap_tensor.cpu().detach()




def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)


def warped_imgs(img,heat,res,fwhm):
    
    img = img.type(torch.float32)
    heat = heat.unsqueeze(dim=1)
    heat = heat.type(torch.float32)
    n = heat.size(0)
    c = heat.size(1)
    h = heat.size(2)
    w = heat.size(3)
    
    f = heat.reshape(n, c, -1)
    norm = nn.Softmax(dim=2)
    # *3 to obtain an appropriate scale for the input of softmax function.
    scale=8
    f_norm = norm(f * scale)
    x = f_norm.view(n,c,h,w)
    
    grid_size = 144,112
    padding_size = 28
    global_size = grid_size[0]+2*padding_size,grid_size[1]+2*padding_size
    

    #We use a gaussian kernel with sigma set to one third of the width of the saliency map 
    gaussian_weights = torch.FloatTensor(makeGaussian(2*padding_size+1, fwhm ))
    #filter = nn.Conv2d(1, 1, kernel_size=(2*padding_size+1,2*padding_size+1),bias=False).to(device = device)
    filter = nn.Conv2d(1, 1, kernel_size=(2*padding_size+1,2*padding_size+1),bias=False)
    #customize convolution kernel weights 
    filter.weight[0].data[:,:,:] = gaussian_weights

    #P_basis = torch.zeros(2,grid_size[0]+2*padding_size,grid_size[1]+2*padding_size, device = device)
    P_basis = torch.zeros(2,grid_size[0]+2*padding_size,grid_size[1]+2*padding_size)
    for k in range(2):
                for i in range(global_size[0]):
                    for j in range(global_size[1]):
                        P_basis[k,i,j] = k*(i-padding_size)/(grid_size[0]-1.0)+(1.0-k)*(j-padding_size)/(grid_size[1]-1.0)


    x = nn.Upsample(size=(grid_size[0],grid_size[1]), mode='bilinear')(x)
    
    x = x #.to(device = device)
    #x = x.view(-1,grid_size[0]*grid_size[1])
    #x = nn.Softmax()(x)*7
    #x = x.view(-1,1,grid_size[0],grid_size[1])
    x = nn.ReplicationPad2d(padding_size)(x)
    #x = nn.ZeroPad2d(padding_size)(x)
    #P = torch.autograd.Variable(torch.zeros(1,2,grid_size[0]+2*padding_size,grid_size[1]+2*padding_size).to(device=device) ,requires_grad=False)
    P = torch.autograd.Variable(torch.zeros(1,2,grid_size[0]+2*padding_size,grid_size[1]+2*padding_size),requires_grad=False)
    P[0,:,:,:] = P_basis
    P = P.expand(x.size(0),2,grid_size[0]+2*padding_size,grid_size[1]+2*padding_size)
    x_cat = torch.cat((x,x),1)
    p_filter = filter(x)
    x_mul = torch.mul(P,x_cat).view(-1,1,global_size[0],global_size[1])
    all_filter = filter(x_mul).view(-1,2,grid_size[0],grid_size[1])
    x_filter = all_filter[:,0,:,:].contiguous().view(-1,1,grid_size[0],grid_size[1])
    y_filter = all_filter[:,1,:,:].contiguous().view(-1,1,grid_size[0],grid_size[1])
    x_filter = x_filter/p_filter
    y_filter = y_filter/p_filter

    xgrids = x_filter*2-1
    ygrids = y_filter*2-1
    xgrids = torch.clamp(xgrids,min=-1,max=1)
    ygrids = torch.clamp(ygrids,min=-1,max=1)

    xgrids = xgrids.view(-1,1,grid_size[0],grid_size[1])
    ygrids = ygrids.view(-1,1,grid_size[0],grid_size[1])

    grid = torch.cat((xgrids,ygrids),1)
    grid = nn.Upsample(size=res, mode='bilinear')(grid)
    grid = torch.transpose(grid,1,2)
    grid = torch.transpose(grid,2,3)
    
    grid = grid.cuda()

    x_sampled = F.grid_sample(img, grid, align_corners = False)

    #img.detach()
    
    return x_sampled #.cpu().detach()
