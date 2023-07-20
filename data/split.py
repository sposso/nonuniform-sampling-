from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os 
import torch
#from ..utils.tools import CBIS_MAMMOGRAM
#Current working directory 


def split_dataset(data_locations):
    '''
    This function splits the dataset paths into training, validation
    and test sets. The new subsets are saved as csv files 
    in the current folder.This function also balances the training data, 
    assigning weights to each class. 
    '''
    #Save here
    folder = os.getcwd()
  

    df = pd.read_csv(data_locations)
    target = df.label.to_numpy()
    train_indices, test_indices = train_test_split(np.arange(target.shape[0]), test_size= 0.15, train_size=0.85, 
                                                   stratify=target, random_state= 42)
    df_train = df.loc[train_indices,:]
    df_test = df.loc[test_indices, :]
    df_test = df_test.reset_index(drop = True)

    if not (os.path.isfile('data/test.csv')):
        df_test.to_csv('data/test.csv', index = False)

    df_train = df_train.reset_index(drop = True )
    label_train = df_train.label.to_numpy()
    train_in, validation_in = train_test_split(np.arange(train_indices.shape[0]), test_size =0.1, train_size = 0.9, 
                                               stratify =label_train, random_state= 42)
    d_train = df_train.loc[train_in, :]
    d_train = d_train.reset_index(drop = True)
    if not (os.path.isfile('data/train.csv')):
        df_test.to_csv('data/train.csv', index = False)

    d_validation = df_train.loc[validation_in,:]
    d_validation = d_validation.reset_index(drop = True)
    if not (os.path.isfile('data/validation.csv')):
        df_test.to_csv('data/train.validation', index = False)


    #####Balancing data to train
    
    class_sample_count = np.unique(df_train.label[train_in], return_counts=True)[1]
    weight = 1. / class_sample_count
    samples_weight = weight[df_train.label[train_in]]
    sampler_weight = torch.from_numpy(samples_weight)

    if not (os.path.isfile("data/sampler_weight.pt")):
        torch.save( sampler_weight, 'data/sampler_weight.pt')
 
    








    





