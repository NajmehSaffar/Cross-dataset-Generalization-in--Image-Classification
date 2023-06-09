import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from dataset import CustomImageDataset
from utils import *
from Baseline2.models import *


class Tester:
    def __init__(self, df_tst_b, df_tst_a, data_dir, setup, exp):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device: ", self.device)
        self.data_dir = data_dir
        self.backup = "/backup01"
        self.setup = setup
        self.exp = exp
        self.num_workers = 4 * torch.cuda.device_count()
        self.num_channels = 1
        self.num_classes = 1
        self.num_datasets = 2
        
        self.im_size = 32
        
        # Build the model
        self.model_voi, self.model_dm, self.model_enc, self.model_dec_h = self.build_model()

        # Load the model
        args = self.load_model()
        
        self.im_size = args.im_size
        self.batch_size = args.batch_size
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize((self.im_size,self.im_size)),
            transforms.Grayscale(num_output_channels=self.num_channels),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
        self.ldr_tstb, self.ldr_tsta = self.set_loader(df_tst_b, df_tst_a)
        
    def set_loader(self, df_test_b, df_test_a):
    
        test_data_b = CustomImageDataset(df_test_b, self.data_dir, self.transform)
        test_loader_b = torch.utils.data.DataLoader(test_data_b, batch_size=self.batch_size, num_workers=self.num_workers, 
                                                    pin_memory=True)
            
        test_data_a = CustomImageDataset(df_test_a, self.data_dir, self.transform)
        test_loader_a = torch.utils.data.DataLoader(test_data_a, batch_size=self.batch_size, num_workers=self.num_workers, 
                                                    pin_memory=True)
        return test_loader_b, test_loader_a 

    
    def build_model(self):
        # Define the model architecture
        
        # ------- Autoencoder -------------------------------
        model_enc = Encoder(im_size=self.im_size, n_channels=self.num_channels).to(self.device)
        model_dec_h = Decoder_h(im_size=self.im_size, n_channels=self.num_channels, n_classes=2).to(self.device)

        # ------- Variable of Interest Classification -------
        model_voi = NaiveNetwork(im_size=self.im_size, n_classes=self.num_classes).to(self.device)

        # ------- Dataset Membership Classification ---------
        model_dm = NaiveNetwork(im_size=self.im_size, n_classes=self.num_datasets).to(self.device) 

        return model_voi, model_dm, model_enc, model_dec_h


    def load_model(self):
        dirname = '../outputs{}/{}/saved_models/'.format(self.backup, self.setup)
        # Assert that the path exists
        assert os.path.exists(dirname), "Path does not exist: {}".format(dirname)

        filename = '{}-{}.pt'.format(self.setup, self.exp)
        # Assert that the file exists within the directory
        assert os.path.isfile(os.path.join(dirname, filename)), "File does not exist: {}".format(filename)

        print('loading the model ...\n') 
        checkpoint = torch.load(dirname + filename)
       
        self.model_enc.load_state_dict(checkpoint['model_enc_state_dict']) 
        self.model_voi.load_state_dict(checkpoint['model_voi_state_dict']) 
        args = checkpoint['args']
        return args
   
        
    def test(self, test_loader):

        # Initialize thelabel, scores and prediction lists(tensors)
        lbllist=torch.zeros(0,dtype=torch.long, device='cpu')
        outlist=torch.zeros(0,dtype=torch.long, device='cpu')
        predlist=torch.zeros(0,dtype=torch.long, device='cpu')

        self.model_voi.eval()
        self.model_dm.eval()
        self.model_enc.eval()
        self.model_dec_h.eval()
        with torch.no_grad():
            for batch, data in enumerate(test_loader):
                
                images, imagenames, labels, dataset_memberships = data
                images, labels = images.to(self.device), labels.to(self.device)
                labels = labels.view(labels.shape[0],1).float()

                Z = self.model_enc(images)
                outputs = self.model_voi(Z)
                
                predicts = torch.round(torch.sigmoid(outputs.data))

                # Append batch prediction results
                lbllist = torch.cat([lbllist, labels.view(-1).cpu()])
                outlist = torch.cat([outlist, outputs.view(-1).cpu()])
                predlist = torch.cat([predlist, predicts.view(-1).cpu()])

        return lbllist, outlist, predlist
    