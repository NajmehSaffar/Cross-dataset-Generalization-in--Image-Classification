import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from dataset import CustomImageDataset
from utils import *
from SupCon.model import *
from SupCon.loss import SupConLoss

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
        
        self.feature_dim = 128             # Output dimension of the projection head
        self.model_type = 'vgg'
        self.model_depth = 16 
        
        self.normal_vec_enc = []
        self.normal_vec_proj = []
        
        # Build the model
        self.model, self.model_head = self.build_model()

        # Load the model
        #args = 
        self.load_model()
        self.im_size = 128 #args.im_size
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
        model = VGG16().to(self.device)
        model_head = ProjectionHead(self.feature_dim, self.model_depth).to(self.device)

        return model, model_head


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
        self.model_head.load_state_dict(checkpoint['model_head_state_dict']) 
        self.normal_vec_enc = checkpoint['normal_vec_enc']
        self.normal_vec_proj = checkpoint['normal_vec_proj']
        #args = checkpoint['args']
        #return args
        
        
    def test(self, test_loader, proj_isTrue=False):

        # Initialize thelabel, scores and prediction lists(tensors)
        lbllist=torch.zeros(0,dtype=torch.long, device='cpu')
        outlist=torch.zeros(0,dtype=torch.long, device='cpu')
        predlist=torch.zeros(0,dtype=torch.long, device='cpu')

        self.model.eval()
        self.model_head.eval()
        with torch.no_grad():
            for batch, data in enumerate(test_loader):
                
                images, imagenames, labels, dataset_memberships = data
                images, labels = images.to(self.device), labels.to(self.device)

                if proj_isTrue:
                    outputs, _ = self.model(images)
                    outputs = self.model_head(outputs)
                    normal_vec = self.normal_vec_proj
                else:
                    _, outputs = self.model(images)
                    normal_vec = self.normal_vec_enc
                
                outputs = outputs.detach()
                similarity_scores = torch.mm(outputs, normal_vec.t())
                predicts = similarity_scores >= 0.5
                
                # Append batch prediction results
                lbllist = torch.cat([lbllist, labels.view(-1).cpu()])
                outlist = torch.cat([outlist, similarity_scores.view(-1).cpu()])
                predlist = torch.cat([predlist, predicts.view(-1).cpu()])

     
        return lbllist, outlist, predlist
    