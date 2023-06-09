import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from dataset import CustomImageDataset
from utils import *

class Trainer:
    def __init__(self, df_trn, df_vld, df_tst_b, df_tst_a, args):
        
        self.device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
        print("Device: ", self.device)
        
        self.num_epochs = args.num_epochs
        self.learning_rate = args.learning_rate
        self.im_size = args.im_size
        self.num_workers = 4 * torch.cuda.device_count()
        self.num_channels = 1
        self.num_classes = 1
        self.momentum = 0.9
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize((self.im_size,self.im_size)),
            transforms.Grayscale(num_output_channels=self.num_channels),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
        
        w_neg = np.sum(df_trn['Beluga_Present']==0) / df_trn.shape[0]
        self.class_weights = torch.FloatTensor([w_neg/(1-w_neg)]).to(self.device)
        
        self.ldr_trn, self.ldr_vld, self.ldr_tstb, self.ldr_tsta = self.set_loader(df_trn, df_vld, df_tst_b, df_tst_a, args)
        
        # Build the model
        self.model = self.build_model().to(self.device)
        
        # Define loss function and optimizer
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        
    def set_loader(self, df_train, df_valid, df_test_b, df_test_a, args):
    
        train_data = CustomImageDataset(df_train, args.data_dir, self.transform)
        valid_data = CustomImageDataset(df_valid, args.data_dir, self.transform)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=self.num_workers, 
                                                   shuffle=True, pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, num_workers=self.num_workers, 
                                                   shuffle=True, pin_memory=True)

        if(args.cross_dataset):

            test_data_b = CustomImageDataset(df_test_b, args.data_dir, self.transform)
            test_data_a = CustomImageDataset(df_test_a, args.data_dir, self.transform)

            test_loader_b = torch.utils.data.DataLoader(test_data_b, batch_size=args.batch_size, num_workers=self.num_workers, 
                                                        shuffle=True, pin_memory=True)
            test_loader_a = torch.utils.data.DataLoader(test_data_a, batch_size=args.batch_size, num_workers=self.num_workers, 
                                                        shuffle=True, pin_memory=True)
        else:
            test_loader_b = None
            test_loader_a = None

        return train_loader, valid_loader, test_loader_b, test_loader_a 

    
    def build_model(self):
        # Define the model architecture
        model = models.vgg16()
        model.features[0] = nn.Conv2d(self.num_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        model.classifier[6] = nn.Linear(4096, self.num_classes)
   
        return model


    def save_model(self, args, train_loss):
        # Create the directory if it doesn't exist
        dirname = '../outputs/{}/saved_models/'.format(args.setup)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            
        filename = '{}-{}.pt'.format(args.setup, args.exp)
        
        print('saving the model ...\n')
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': train_loss,
            'args': args
        }
        torch.save(state, dirname + filename)
        del state
   
        
    def train(self, train_loader, args):
        
        train_loss = []
        for epoch in range(self.num_epochs):
            
            train_epoch_loss = AverageMeter()
            self.model.train()
            for batch, data in enumerate(train_loader):
                
                images, imagenames, labels, dataset_memberships = data
                images, labels = images.to(self.device), labels.to(self.device)
                labels = labels.view(labels.shape[0],1).float()

                self.optimizer.zero_grad()
                outputs = self.model(images)
                
                # Calculate the batch's loss
                loss = self.criterion(outputs, labels) 
                loss.backward()
                self.optimizer.step()

                # Update the epoch's loss 
                train_epoch_loss.update(loss.item(), 1)

            train_loss.append(train_epoch_loss.avg)
            print(f'Training Process is running: Epoch [{epoch + 1}/{self.num_epochs}], Loss: {train_epoch_loss.avg}')

        # Create the directory if it doesn't exist
        dirname = '../outputs/{}/figures/'.format(args.setup)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        filename = '-{}-{}'.format(args.setup, args.exp)
        
        plot_loss([train_loss], ['train loss'], ['green'], dirname, filename)
        
        self.save_model(args, train_loss)
        
        
        
    def validate(self, valid_loader, args, name_of_set):
        correct = 0
        total = 0

        # Initialize thelabel, scores and prediction lists(tensors)
        lbllist=torch.zeros(0,dtype=torch.long, device='cpu')
        outlist=torch.zeros(0,dtype=torch.long, device='cpu')
        predlist=torch.zeros(0,dtype=torch.long, device='cpu')

        self.model.eval()
        with torch.no_grad():
            for batch, data in enumerate(valid_loader):
                
                images, imagenames, labels, dataset_memberships = data
                images, labels = images.to(self.device), labels.to(self.device)
                labels = labels.view(labels.shape[0],1).float()

                outputs = self.model(images)
                
                predicts = torch.round(torch.sigmoid(outputs.data))

                total += outputs.size(0)
                correct += torch.eq(predicts, labels).sum().double().item()

                # Append batch prediction results
                lbllist = torch.cat([lbllist, labels.view(-1).cpu()])
                outlist = torch.cat([outlist, outputs.view(-1).cpu()])
                predlist = torch.cat([predlist, predicts.view(-1).cpu()])

        # Create the directory if it doesn't exist
        dirname = '../outputs/{}/figures/'.format(args.setup)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        filename = '-{}-{}-{}'.format(args.setup, args.exp, name_of_set)
        
        #print('\nAccuracy of the network on test images: %d %%\n' % (100 * correct / total))
        plot_CONFMAT(lbllist.numpy(), predlist.numpy(), dirname, filename)
        roc_auc = plot_ROC(lbllist.numpy(), outlist.numpy(), dirname, filename)
        plot_histogram(lbllist, outlist, dirname, filename)

        return roc_auc
    