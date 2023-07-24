import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from dataset import CustomImageDataset
from utils import *
from Baseline2.models import *
from Baseline2.proportional_batch_sampler import ProportionalBatchSampler


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
        self.num_datasets = 2
        self.momentum = 0.9
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize((self.im_size,self.im_size)),
            transforms.Grayscale(num_output_channels=self.num_channels),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
        self.parameters = {"beta":1.0, "gamma":1.0, "m":1, "n":1}
        
        P_d0_c0 = np.sum((df_trn["dataset_membership"]==0) & (df_trn["Beluga_Present"]==0)) / len(df_trn)
        P_d0_c1 = np.sum((df_trn["dataset_membership"]==0) & (df_trn["Beluga_Present"]==1)) / len(df_trn)
        P_d1_c0 = np.sum((df_trn["dataset_membership"]==1) & (df_trn["Beluga_Present"]==0)) / len(df_trn)
        P_d1_c1 = np.sum((df_trn["dataset_membership"]==1) & (df_trn["Beluga_Present"]==1)) / len(df_trn)
        self.proportions = [P_d0_c0, P_d0_c1, P_d1_c0, P_d1_c1]
        
        self.ldr_trn, self.ldr_vld, self.ldr_tstb, self.ldr_tsta = self.set_loader(df_trn, df_vld, df_tst_b, df_tst_a, args)
        
        # Build the model
        self.model_voi, self.model_dm, self.model_enc, self.model_dec_h = self.build_model()
        
        # Define loss functions
        self.criterion_voi = nn.BCEWithLogitsLoss()
        self.criterion_dm = nn.CrossEntropyLoss()#reduction='none')
        self.criterion_dec = nn.MSELoss()
    
        # Define optimizer
        self.optimizer = torch.optim.Adam([
            {'params': self.model_voi.parameters()},
            {'params': self.model_dm.parameters()},
            {'params': self.model_enc.parameters(), 'lr': 1e-4},
            {'params': self.model_dec_h.parameters(), 'lr': 1e-4}
            ], lr=self.learning_rate)
        
        
    def set_loader(self, df_train, df_valid, df_test_b, df_test_a, args):
    
        train_data = CustomImageDataset(df_train, args.data_dir, self.transform)
        valid_data = CustomImageDataset(df_valid, args.data_dir, self.transform)
        test_data_b = CustomImageDataset(df_test_b, args.data_dir, self.transform)
        test_data_a = CustomImageDataset(df_test_a, args.data_dir, self.transform)
        
        sampler = ProportionalBatchSampler(train_data, args.batch_size, self.proportions)
        train_loader = torch.utils.data.DataLoader(train_data, batch_sampler=sampler, num_workers=self.num_workers, 
                                                   pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, num_workers=self.num_workers, 
                                                   shuffle=True, pin_memory=True)
        test_loader_b = torch.utils.data.DataLoader(test_data_b, batch_size=args.batch_size, num_workers=self.num_workers,
                                                    shuffle=True, pin_memory=True)
        test_loader_a = torch.utils.data.DataLoader(test_data_a, batch_size=args.batch_size, num_workers=self.num_workers, 
                                                    shuffle=True, pin_memory=True)
        return train_loader, valid_loader, test_loader_b, test_loader_a 

    
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
    

    def save_model(self, args, train_loss):
        # Create the directory if it doesn't exist
        dirname = '../outputs/{}/saved_models/'.format(args.setup)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            
        filename = '{}-{}.pt'.format(args.setup, args.exp)
        
        print('saving the model ...\n')
        state = {
            'model_voi_state_dict': self.model_voi.state_dict(),
            'model_dm_state_dict': self.model_dm.state_dict(),
            'model_enc_state_dict': self.model_enc.state_dict(),
            'model_dec_h_state_dict': self.model_dec_h.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': train_loss,
            'args': args
        }
        torch.save(state, dirname + filename)
        del state
   
    
    def set_grad_flags(self, flag_voi=False, flag_dm=False, flag_enc=False, flag_dec_h=False):
        for param in self.model_voi.parameters():
            param.requires_grad = flag_voi

        for param in self.model_dm.parameters():
            param.requires_grad = flag_dm

        for param in self.model_enc.parameters():
            param.requires_grad = flag_enc

        for param in self.model_dec_h.parameters():
            param.requires_grad = flag_dec_h
        
        
    def train(self, train_loader, args):
        
        train_loss, train_loss_voi, train_loss_dm, train_loss_h = [], [], [], []
    
        for epoch in range(self.num_epochs):
            
            train_epoch_loss_voi = AverageMeter()
            train_epoch_loss_dm = AverageMeter()
            train_epoch_loss_h = AverageMeter()
            train_epoch_loss = AverageMeter()

            self.model_voi.train()
            self.model_dm.train()
            self.model_enc.train()
            self.model_dec_h.train()
            for batch, data in enumerate(train_loader):
                
                images, imagenames, labels, dataset_memberships = data
                images, labels, A = images.to(self.device), labels.to(self.device), dataset_memberships.to(self.device)
                labels = labels.view(labels.shape[0],1).float()
                
                # =================== Backward variable_of_interest ====================
                self.set_grad_flags(flag_voi=True, flag_dm=False, flag_enc=True, flag_dec_h=True)
                Z = self.model_enc(images)
                X_h = self.model_dec_h(Z, A)
                outputs_Y = self.model_voi(Z)
                outputs_A = self.model_dm(Z)
        
                # Calculate the batch's loss
                loss_h = self.criterion_dec(X_h, images)
                loss_voi = self.criterion_voi(outputs_Y, labels)
                loss_dm = self.criterion_dm(outputs_A, A.long()) #loss_dm = torch.sum((loss_dm * wts))
                loss = (loss_h 
                        + self.parameters.get('beta', 1.0) * loss_voi 
                        - self.parameters.get('gamma', 1.0) * loss_dm)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update the epoch's loss 
                train_epoch_loss_voi.update(loss_voi.item(), 1)
                train_epoch_loss_h.update(loss_h.item(), 1)
                train_epoch_loss.update(loss.item(), 1)
                
                
                 # =================== Backward dataset_membership ====================
                for i in range(5):
                    
                    self.set_grad_flags(flag_voi=False, flag_dm=True, flag_enc=False, flag_dec_h=False)
                    Z = self.model_enc(images)
                    X_h = self.model_dec_h(Z, A)
                    outputs_Y = self.model_voi(Z)
                    outputs_A = self.model_dm(Z)

                    # Calculate the batch's loss
                    loss_h = self.criterion_dec(X_h, images)
                    loss_voi = self.criterion_voi(outputs_Y, labels)
                    loss_dm = self.criterion_dm(outputs_A, A.long()) #loss_dm = torch.sum((loss_dm * wts))
                    loss = -(loss_h 
                            + self.parameters.get('beta', 1.0) * loss_voi 
                            - self.parameters.get('gamma', 1.0) * loss_dm)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Update meters 
                    train_epoch_loss_dm.update(loss_dm.item(), 1)
                
            train_loss.append(train_epoch_loss.avg)
            train_loss_voi.append(train_epoch_loss_voi.avg)
            train_loss_dm.append(train_epoch_loss_dm.avg)
            train_loss_h.append(train_epoch_loss_h.avg)
        
            print(f'Training Process is running: Epoch [{epoch + 1}/{self.num_epochs}], Total Loss: {train_epoch_loss.avg} \
            | variable_of_interest loss: {train_epoch_loss_voi.avg} | dataset_membership loss: {train_epoch_loss_dm.avg} \
            | dec_h loss: {train_epoch_loss_h.avg} ')
    
        # Create the directory if it doesn't exist
        dirname = '../outputs/{}/figures/'.format(args.setup)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        filename = '-{}-{}'.format(args.setup, args.exp)
        
        plot_loss([train_loss, train_loss_voi, train_loss_dm, train_loss_h], 
              ['combined loss', 'variable_of_interest_C loss', 'dataset_membership_C loss', 'dec_h loss'], 
              ['green', 'blue', 'red', 'orange'], dirname, filename)
        
        plot_loss([train_loss], ['combined loss'], ['green'], dirname, filename)
        plot_loss([train_loss_voi], ['variable_of_interest_C loss'], ['blue'], dirname, filename)
        plot_loss([train_loss_dm], ['dataset_membership_C loss'], ['red'], dirname, filename)
        plot_loss([train_loss_h], ['dec_h loss'], ['orange'], dirname, filename)
    
        self.save_model(args, train_loss)
        
        
    def validate(self, valid_loader, args, name_of_set):
        correct = 0
        total = 0

        # Initialize thelabel, scores and prediction lists(tensors)
        lbllist=torch.zeros(0,dtype=torch.long, device='cpu')
        outlist=torch.zeros(0,dtype=torch.long, device='cpu')
        predlist=torch.zeros(0,dtype=torch.long, device='cpu')

        self.model_voi.eval()
        self.model_dm.eval()
        self.model_enc.eval()
        self.model_dec_h.eval()
        with torch.no_grad():
            for batch, data in enumerate(valid_loader):
                
                images, imagenames, labels, dataset_memberships = data
                images, labels = images.to(self.device), labels.to(self.device)
                labels = labels.view(labels.shape[0],1).float()

                Z = self.model_enc(images)
                outputs = self.model_voi(Z)
                
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
        plot_histogram(lbllist.numpy(), outlist.numpy(), dirname, filename)

        return roc_auc
    