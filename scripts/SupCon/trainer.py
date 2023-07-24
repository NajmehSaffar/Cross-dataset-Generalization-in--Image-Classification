import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from dataset import CustomImageDataset
from utils import *
from SupCon.model import *
from SupCon.loss import SupConLoss

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
        self.weight_decay = 1e-1
        self.dampening = 0.0
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize((self.im_size,self.im_size)),
            transforms.Grayscale(num_output_channels=self.num_channels),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
        self.feature_dim = 128             # Output dimension of the projection head
        self.model_type = 'vgg'
        self.model_depth = 16 
        self.K = 8                         # 8,16,32
        
        self.memory_bank_size = 1000
        self.temp = 0.9                    # Temperature for loss function
            
        self.n_train_batch_size = int(args.batch_size * (1         /self.K))  # Batch Size for normal (pos) training data
        self.a_train_batch_size = int(args.batch_size * ((self.K-1)/self.K))  # Batch Size for anormal (neg) training data
        
        self.normal_vec_enc = []
        self.normal_vec_proj = []
        
        self.ldr_trn, self.ldr_trn_neg, self.ldr_trn_pos, self.ldr_vld, self.ldr_tstb, self.ldr_tsta = self.set_loader(
            df_trn, df_vld, df_tst_b, df_tst_a, args)
        
        # Build the model
        self.model, self.model_head = self.build_model()
        
        # Define loss function and optimizer
        self.criterion = SupConLoss()
        self.optimizer = torch.optim.SGD([
            {'params': self.model.parameters()}, 
            {'params': self.model_head.parameters()}
            ], lr=self.learning_rate, momentum=self.momentum, dampening=self.dampening, weight_decay=self.weight_decay)
        

    def set_loader(self, df_train, df_valid, df_test_b, df_test_a, args):
    
        train_data = CustomImageDataset(df_train, args.data_dir, self.transform)
        valid_data = CustomImageDataset(df_valid, args.data_dir, self.transform)
        test_data_b = CustomImageDataset(df_test_b, args.data_dir, self.transform)
        test_data_a = CustomImageDataset(df_test_a, args.data_dir, self.transform)
        
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=self.num_workers, 
                                                   shuffle=True, pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, num_workers=self.num_workers, 
                                                   shuffle=True, pin_memory=True)
        test_loader_b = torch.utils.data.DataLoader(test_data_b, batch_size=args.batch_size, num_workers=self.num_workers,
                                                    shuffle=True, pin_memory=True)
        test_loader_a = torch.utils.data.DataLoader(test_data_a, batch_size=args.batch_size, num_workers=self.num_workers, 
                                                    shuffle=True, pin_memory=True)
        
        df_train_neg = df_train[df_train['Beluga_Present'] == 0]
        df_train_pos = df_train[df_train['Beluga_Present'] == 1]
        train_neg_dataset = CustomImageDataset(df_train_neg, args.data_dir, self.transform)
        train_pos_dataset = CustomImageDataset(df_train_pos, args.data_dir, self.transform)
        
        train_neg_loader = torch.utils.data.DataLoader(train_neg_dataset, self.a_train_batch_size, num_workers=self.num_workers,
                                                       shuffle=True, pin_memory=True)
        train_pos_loader = torch.utils.data.DataLoader(train_pos_dataset, self.n_train_batch_size, num_workers=self.num_workers,
                                                       shuffle=True, pin_memory=True)
        return train_loader, train_neg_loader, train_pos_loader, valid_loader, test_loader_b, test_loader_a 
    
    
    def build_model(self):
        # Define the model architecture
        model = VGG16().to(self.device)
        model_head = ProjectionHead(self.feature_dim, self.model_depth).to(self.device)

        return model, model_head


    def save_model(self, args, train_loss):
        # Create the directory if it doesn't exist
        dirname = '../outputs/{}/saved_models/'.format(args.setup)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            
        filename = '{}-{}.pt'.format(args.setup, args.exp)
        
        print('saving the model ...\n')
        state = {
            'model_state_dict': self.model.state_dict(),
            'model_head_state_dict': self.model_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': train_loss,
            'args': args,
            'normal_vec_enc': self.normal_vec_enc,
            'normal_vec_proj': self.normal_vec_proj
        }
        torch.save(state, dirname + filename)
        del state
   
        
    def train(self, train_loader, args):
        train_neg_loader, train_pos_loader = train_loader
        
        memory_bank_1 = []
        memory_bank_2 = []
        train_loss = []
        for epoch in range(self.num_epochs):
            
            train_epoch_loss = AverageMeter()
            self.model.train()
            self.model_head.train()
            for batch, ((neg_data), (pos_data)) in enumerate(zip(train_neg_loader, train_pos_loader)):

                neg_data, pos_data = neg_data[0].to(self.device), pos_data[0].to(self.device)
                data = torch.cat((pos_data, neg_data), dim=0)   # n_vec as well as a_vec are all normalized value

                self.optimizer.zero_grad()
                
                unnormed_vec, normed_vec = self.model(data)
                    
                vec = self.model_head(unnormed_vec) 
                pos_vec = vec[0:pos_data.shape[0]]            # K normal (pos) 8
                neg_vec = vec[pos_data.shape[0]:]             # M anomalous (neg) 56
                
                loss = self.criterion(pos_vec, neg_vec, self.temp) # Calculate the batch's loss
                loss.backward()
                self.optimizer.step()

                # ===========update memory bank===============
                self.model.eval()
                self.model_head.eval()
                with torch.no_grad():
                    unnormed_vec, normed_enc_vec = self.model(pos_data)
                    normed_proj_vec = self.model_head(unnormed_vec)

                    average_enc = torch.mean(normed_enc_vec, dim=0, keepdim=True)
                    average_proj = torch.mean(normed_proj_vec, dim=0, keepdim=True)

                    if len(memory_bank_1) > self.memory_bank_size:
                        memory_bank_1.pop(0)
                        memory_bank_2.pop(0)

                    memory_bank_1.append(average_enc) 
                    memory_bank_2.append(average_proj)
            
                self.model.train() # Turn back to training mode after eval step
                self.model_head.train()   

                # ===============update meters ===============
                train_epoch_loss.update(loss.item(), 1)

            train_loss.append(train_epoch_loss.avg)
            print(f'Training Process is running: Epoch [{epoch + 1}/{self.num_epochs}], Loss: {train_epoch_loss.avg}')

            self.normal_vec_enc = torch.mean(torch.cat(memory_bank_1, dim=0), dim=0, keepdim=True)
            self.normal_vec_enc = l2_normalize(self.normal_vec_enc)

            self.normal_vec_proj = torch.mean(torch.cat(memory_bank_2, dim=0), dim=0, keepdim=True)
            self.normal_vec_proj = l2_normalize(self.normal_vec_proj)
            
        # Create the directory if it doesn't exist
        dirname = '../outputs/{}/figures/'.format(args.setup)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        filename = '-{}-{}'.format(args.setup, args.exp)
        
        plot_loss([train_loss], ['train loss'], ['green'], dirname, filename, args.display)
        
        self.save_model(args, train_loss)
        
        
    def validate(self, valid_loader, normal_vec, args, name_of_set, proj_isTrue=False):

        # Initialize thelabel, scores and prediction lists(tensors)
        lbllist=torch.zeros(0,dtype=torch.long, device='cpu')
        outlist=torch.zeros(0,dtype=torch.long, device='cpu')
        predlist=torch.zeros(0,dtype=torch.long, device='cpu')

        self.model.eval()
        self.model_head.eval()
        with torch.no_grad():
            for batch, data in enumerate(valid_loader):
                
                images, imagenames, labels, dataset_memberships = data
                images, labels = images.to(self.device), labels.to(self.device)

                if proj_isTrue:
                    outputs, _ = self.model(images)
                    outputs = self.model_head(outputs)
                else:
                    _, outputs = self.model(images)
                
                outputs = outputs.detach()
                similarity_scores = torch.mm(outputs, normal_vec.t())
                predicts = similarity_scores >= 0.5
                
                # Append batch prediction results
                lbllist = torch.cat([lbllist, labels.view(-1).cpu()])
                outlist = torch.cat([outlist, similarity_scores.view(-1).cpu()])
                predlist = torch.cat([predlist, predicts.view(-1).cpu()])

        # Create the directory if it doesn't exist
        dirname = '../outputs/{}/figures/'.format(args.setup)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        filename = '-{}-{}-{}'.format(args.setup, args.exp, name_of_set)
        
        #print('\nAccuracy of the network on test images: %d %%\n' % (100 * correct / total))
        plot_CONFMAT(lbllist.numpy(), predlist.numpy(), dirname, filename, args.display)
        roc_auc = plot_ROC(lbllist.numpy(), outlist.numpy(), dirname, filename, args.display)
        plot_histogram(lbllist.numpy(), outlist.numpy(), dirname, filename, args.display)

        return roc_auc
    