import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, dataframe, data_dir, transform = None):
        # Load and preprocess the data
        self.dataframe = dataframe
        self.data_dir = data_dir
        self.transform = transform
        
        self.dir_names = dataframe['dir_name'].values.tolist()
        self.img_names = dataframe['file_name'].values.tolist()
        self.Y = dataframe['Beluga_Present'].values.tolist() #Variable of Interest (0/1)
        self.A = dataframe['dataset_membership'].values.tolist() #Dataset Membership (0/1/2)

        
    def __len__(self):
        # Return the total number of samples in the dataset    
        return len(self.Y)

    
    def __getitem__(self, idx):
        # Retrieve and preprocess a single sample from the dataset
        
        y = self.Y[idx]
        a = self.A[idx]
        
        img = Image.open(os.path.join(self.data_dir, self.dir_names[idx], self.img_names[idx])) 
        if self.transform is not None:
            img = self.transform(img)
            
        imgname = self.img_names[idx]
        
        return img, imgname, y, a
    
    
# Load the dataframe from a file  
def load_data(args):
    
    print('\nloading the data ...\n')

    # ------------------------------------- Set 1 -------------------------------------
    df1 = pd.read_csv(os.path.join(args.data_dir, "Good Quality Photos List.csv")) 
    df1['dataset_membership'] = 0
    df1['dir_name'] = "Good_Quality_Photos"
    n_images = len(df1['file_name'])
    
    indices_a, indices_b = split_indices(np.arange(n_images), [0.5, 0.5])
    df1_a = df1.loc[df1.index.isin(indices_a)]
    df1_b = df1.loc[df1.index.isin(indices_b)]
    # Assert if intersection is empty
    intersection = pd.merge(df1_a, df1_b, how='inner', on=['file_name'])
    assert intersection.empty, "Intersection of dataframes is not empty."
    
    indices_trn, indices_vld = split_indices(np.arange(n_images), [0.9, 0.1])
    df1_trn = df1.loc[df1.index.isin(indices_trn)]
    df1_vld = df1.loc[df1.index.isin(indices_vld)]
    # Assert if intersection is empty
    intersection = pd.merge(df1_trn, df1_vld, how='inner', on=['file_name'])
    assert intersection.empty, "Intersection of dataframes is not empty."
    
    
    # ------------------------------------- Set 2 -------------------------------------
    df2 = pd.read_csv(os.path.join(args.data_dir, "Bad Quality Photos List.csv")) 
    df2['dataset_membership'] = 1 
    df2['dir_name'] = "Bad_Quality_Photos"
    n_images = len(df2['file_name'])
   
    indices_a, indices_b = split_indices(np.arange(n_images), [0.5, 0.5])
    df2_a = df2.loc[df2.index.isin(indices_a)]
    df2_b = df2.loc[df2.index.isin(indices_b)]
    # Assert if intersection is empty
    intersection = pd.merge(df2_a, df2_b, how='inner', on=['file_name'])
    assert intersection.empty, "Intersection of dataframes is not empty."
    
    indices_trn, indices_vld = split_indices(np.arange(n_images), [0.9, 0.1])
    df2_trn = df2.loc[df2.index.isin(indices_trn)]
    df2_vld = df2.loc[df2.index.isin(indices_vld)]
    # Assert if intersection is empty
    intersection = pd.merge(df2_trn, df2_vld, how='inner', on=['file_name'])
    assert intersection.empty, "Intersection of dataframes is not empty."
    
    
    # ------------------------------------- Set 3 -------------------------------------
    df3 = pd.read_csv(os.path.join(args.data_dir, "Half Half Photos List.csv")) 
    df3['dataset_membership'] = 2
    df3['dir_name'] = "Half_Half_Photos"
    n_images = len(df3['file_name'])
    
    indices_a, indices_b = split_indices(np.arange(n_images), [0.5, 0.5])
    df3_a = df3.loc[df3.index.isin(indices_a)]
    df3_b = df3.loc[df3.index.isin(indices_b)]
    # Assert if intersection is empty
    intersection = pd.merge(df3_a, df3_b, how='inner', on=['file_name'])
    assert intersection.empty, "Intersection of dataframes is not empty."
    
    indices_trn, indices_vld = split_indices(np.arange(n_images), [0.9, 0.1])
    df3_trn = df3.loc[df3.index.isin(indices_trn)]
    df3_vld = df3.loc[df3.index.isin(indices_vld)]
    # Assert if intersection is empty
    intersection = pd.merge(df3_trn, df3_vld, how='inner', on=['file_name'])
    assert intersection.empty, "Intersection of dataframes is not empty."
    
    
    print("Set1 nfiles:", len(df1), ",  Set2 nfiles:", len(df2), ",  Set3 nfiles:", len(df3))
    print('\ndone ...\n')
    
    return [df1_trn, df1_vld, df1_a, df1_b], [df2_trn, df2_vld, df2_a, df2_b], [df3_trn, df3_vld, df3_a, df3_b]



def split_indices(indices, split_percentages, shuffle=True, seed=30):
    
    assert sum(split_percentages) == 1, "The sum of percentages is not equal to 1."
    
    np.random.seed(seed)
    if shuffle:
        np.random.shuffle(indices)
    
    total_samples = len(indices)
    splits = []
    start_idx = 0
    
    for percentage in split_percentages:
        size = int(percentage * total_samples)
        end_idx = start_idx + size
        splits.append(indices[start_idx:end_idx])
        start_idx = end_idx
    
    return splits