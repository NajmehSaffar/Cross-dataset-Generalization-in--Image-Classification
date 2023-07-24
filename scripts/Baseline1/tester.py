import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from dataset import CustomImageDataset
from utils import *

class Tester:
    def __init__(self, df_tst_b, df_tst_a, data_dir, backup, setup, exp):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device: ", self.device)
        self.data_dir = data_dir
        self.backup = backup
        self.setup = setup
        self.exp = exp
        self.num_workers = 4 * torch.cuda.device_count()
        self.num_channels = 1
        self.num_classes = 1
        
        # Build the model
        self.model = self.build_model().to(self.device)

        # Load the model
        args = self.load_model()
        
        self.im_size = 32 #args.im_size
        self.batch_size = 256 #args.batch_size
        
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
        model = models.vgg16()
        model.features[0] = nn.Conv2d(self.num_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        model.classifier[6] = nn.Linear(4096, self.num_classes)
   
        return model


    def load_model(self):
        dirname = '../outputs{}/{}/saved_models/'.format(self.backup, self.setup)
        # Assert that the path exists
        assert os.path.exists(dirname), "Path does not exist: {}".format(dirname)

        filename = '{}-{}.pt'.format(self.setup, self.exp)
        # Assert that the file exists within the directory
        assert os.path.isfile(os.path.join(dirname, filename)), "File does not exist: {}".format(filename)

        print('loading the model ...\n') 
        checkpoint = torch.load(dirname + filename)
        
        self.model.load_state_dict(checkpoint['model_state_dict']) 
        #args = checkpoint['args']
        #return args
   
        
    def test(self, test_loader):

        # Initialize thelabel, scores and prediction lists(tensors)
        lbllist=torch.zeros(0,dtype=torch.long, device='cpu')
        outlist=torch.zeros(0,dtype=torch.long, device='cpu')
        predlist=torch.zeros(0,dtype=torch.long, device='cpu')

        self.model.eval()
        with torch.no_grad():
            for batch, data in enumerate(test_loader):
                
                images, imagenames, labels, dataset_memberships = data
                images, labels = images.to(self.device), labels.to(self.device)
                labels = labels.view(labels.shape[0],1).float()

                outputs = self.model(images)
                
                predicts = torch.round(torch.sigmoid(outputs.data))

                # Append batch prediction results
                lbllist = torch.cat([lbllist, labels.view(-1).cpu()])
                outlist = torch.cat([outlist, outputs.view(-1).cpu()])
                predlist = torch.cat([predlist, predicts.view(-1).cpu()])

        return lbllist, outlist, predlist
    