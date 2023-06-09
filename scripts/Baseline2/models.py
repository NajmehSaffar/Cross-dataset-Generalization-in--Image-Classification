import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)

class Encoder(nn.Module):
    def __init__(self, im_size, n_channels):
        super(Encoder, self).__init__()
        self.height, self.width, self.nchannel = im_size, im_size, n_channels
        self.ksize, self.z_dim = 3, 512
        
        # The padding is calculated as kernel_size // 2, which ensures that 
        # the output size remains the same as the input size. 
        self.padding = self.ksize // 2
        
        i = 3 # Number of (conv + maxpool) layers
        last_channel_size = 32
        
        # Encoder Layers
        self.enc_bn1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(in_channels=self.nchannel, out_channels=8, kernel_size=self.ksize, stride=1,
                               padding=self.padding) 

        self.enc_bn2 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=self.ksize, stride=1,
                               padding=self.padding)

        self.enc_bn3 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=last_channel_size, kernel_size=self.ksize, stride=1,
                               padding=self.padding)
        
        if (self.height == 128 & self.width == 128):
            i = i + 2 # Number of (conv + maxpool) layers
            last_channel_size = 128
            
            self.enc_bn4 = nn.BatchNorm2d(32)
            self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=self.ksize, stride=1,
                                   padding=self.padding)

            self.enc_bn5 = nn.BatchNorm2d(64)
            self.conv5 = nn.Conv2d(in_channels=64, out_channels=last_channel_size, kernel_size=self.ksize, stride=1,
                                   padding=self.padding)

        self.flat = Flatten()

        self.en_linear1 = nn.Linear((self.height//(2**i)) * (self.width//(2**i)) * last_channel_size, 256) # W * H * C => 256
        self.en_linear2 = nn.Linear(256, self.z_dim)
        
        # Other Layers
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool2d(2)
        
    def forward(self,x):
        x = self.enc_bn1(x)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.maxpool(x)

        x = self.enc_bn2(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.maxpool(x)

        x = self.enc_bn3(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.maxpool(x)
        
        if (self.height == 128 & self.width == 128):
            x = self.enc_bn4(x)
            x = self.conv4(x)
            x = self.leaky_relu(x)
            x = self.maxpool(x)

            x = self.enc_bn5(x)
            x = self.conv5(x)
            x = self.leaky_relu(x)
            x = self.maxpool(x)
        
        x = self.flat(x)
        
        x = self.en_linear1(x)
        x = self.leaky_relu(x)
        x = self.en_linear2(x)

        return (x)

            
class Decoder_g(nn.Module):
    def __init__(self, im_size, n_channels):
        super(Decoder_g, self).__init__()
        self.height, self.width, self.nchannel = im_size, im_size, n_channels
        self.ksize, self.z_dim = 3, 512
        self.padding = self.ksize // 2
        
        self.nLayers = 3 # Number of (conv + maxpool) layers
        self.last_channel_size = 32
        
        if (self.height == 128 & self.width == 128):
            self.nLayers = self.nLayers + 2 # Number of (conv + maxpool) layers
            self.last_channel_size = 128
            
        # Decoder Layers
        self.dec_linear2 = nn.Linear(self.z_dim, 256)
        self.dec_linear1 = nn.Linear(256, (self.height//(2**self.nLayers)) * (self.width//(2**self.nLayers)) * 
                                     self.last_channel_size)

        if (self.height == 128 & self.width == 128):
            self.dec_bn5 = nn.BatchNorm2d(128)
            self.convT5 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=self.ksize, stride=1,
                                             padding=self.padding)
            
            self.dec_bn4 = nn.BatchNorm2d(64)
            self.convT4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.ksize, stride=1,
                                             padding=self.padding)

            
        self.dec_bn3 = nn.BatchNorm2d(32)
        self.convT3 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=self.ksize, stride=1,
                                         padding=self.padding)

        self.dec_bn2 = nn.BatchNorm2d(16)
        self.convT2 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=self.ksize, stride=1,
                                         padding=self.padding)

        self.dec_bn1 = nn.BatchNorm2d(8)
        self.convT1 = nn.ConvTranspose2d(in_channels=8, out_channels=self.nchannel, kernel_size=self.ksize, stride=1,
                                         padding=self.padding)

        # Other Layers
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self,x):
        x = self.dec_linear2(x)
        x = self.leaky_relu(x)
        x = self.dec_linear1(x)

        x = x.reshape(x.size(0), self.last_channel_size, (self.height//(2**self.nLayers)), (self.width//(2**self.nLayers)))

        if (self.height == 128 & self.width == 128):
            x = self.dec_bn5(x)
            x = self.convT5(x)
            x = self.leaky_relu(x)
            x = self.upsample(x)

            x = self.dec_bn4(x)
            x = self.convT4(x)
            x = self.leaky_relu(x)
            x = self.upsample(x)
            
            
        x = self.dec_bn3(x)
        x = self.convT3(x)
        x = self.leaky_relu(x)
        x = self.upsample(x)

        x = self.dec_bn2(x)
        x = self.convT2(x)
        x = self.leaky_relu(x)
        x = self.upsample(x)

        x = self.dec_bn1(x)
        x = self.convT1(x)
        x = self.leaky_relu(x)
        x = self.upsample(x)

        return x


class Decoder_h(nn.Module):
    def __init__(self, im_size, n_channels, n_classes):
        super(Decoder_h, self).__init__()
        self.height, self.width, self.nchannel, self.nclasses = im_size, im_size, n_channels, n_classes
        self.ksize, self.z_dim = 3, 512
        self.padding = self.ksize // 2
        
        self.nLayers = 3 # Number of (conv + maxpool) layers
        self.last_channel_size = 32
        
        if (self.height == 128 & self.width == 128):
            self.nLayers = self.nLayers + 2 # Number of (conv + maxpool) layers
            self.last_channel_size = 128
            
        # Decoder Layers
        self.dec_linear2 = nn.Linear(self.z_dim, 256)
        self.dec_linear1 = nn.Linear(256, (self.height//(2**self.nLayers)) * (self.width//(2**self.nLayers)) * 
                                     self.last_channel_size)

        if (self.height == 128 & self.width == 128):
            self.dec_bn5 = nn.BatchNorm2d(128)
            self.convT5 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=self.ksize, stride=1,
                                             padding=self.padding)
            
            self.dec_bn4 = nn.BatchNorm2d(64)
            self.convT4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.ksize, stride=1,
                                             padding=self.padding)

            
        self.dec_bn3 = nn.BatchNorm2d(32)
        self.convT3 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=self.ksize, stride=1,
                                         padding=self.padding)

        self.dec_bn2 = nn.BatchNorm2d(16)
        self.convT2 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=self.ksize, stride=1,
                                         padding=self.padding)

        self.dec_bn1 = nn.BatchNorm2d(8)
        self.convT1 = nn.ConvTranspose2d(in_channels=8, out_channels=self.nchannel, kernel_size=self.ksize, stride=1,
                                         padding=self.padding)
        
        # Other Layers
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.flat = Flatten()
        
        self.conv1x1 = nn.Conv1d(in_channels=(1+self.nclasses), out_channels=8, kernel_size=1)
        self.mid_linear = nn.Linear(self.z_dim*8, self.z_dim)
        
    def forward(self, x, dataset_membership):
        # -------------------- Concatenate dataset_membership to x --------------------
        dataset_membership = dataset_membership.reshape(len(dataset_membership), 1)
        
        one_hot_target = (dataset_membership == torch.arange(self.nclasses).reshape(1, self.nclasses).cuda()).float() # B_Size x num_classes
        one_hot_target = one_hot_target[:, :, None]                             # B_Size x num_classes x 1
        one_hot_target = one_hot_target.repeat(1, 1, x.shape[1])                # B_Size x num_classes x num_of_features
        
        x = x[:, None, :]                                                       # B_Size x 1 x num_of_features
        x = torch.cat((x, one_hot_target), 1)                                   # B_Size x (1 + num_classes) x num_of_features   
        
        x = self.conv1x1(x)                                                     # B_Size x 10 x num_of_features
        x = self.leaky_relu(x)
        x = self.flat(x)                                                        # B_Size x (10*num_of_features)
        x = self.mid_linear(x)                                                  # B_Size x (num_of_features)
        
        # ---------------------------------- Decoder ----------------------------------
        x = self.dec_linear2(x)
        x = self.leaky_relu(x)
        x = self.dec_linear1(x)

        x = x.reshape(x.size(0), self.last_channel_size, (self.height//(2**self.nLayers)), (self.width//(2**self.nLayers)))

        if (self.height == 128 & self.width == 128):
            x = self.dec_bn5(x)
            x = self.convT5(x)
            x = self.leaky_relu(x)
            x = self.upsample(x)

            x = self.dec_bn4(x)
            x = self.convT4(x)
            x = self.leaky_relu(x)
            x = self.upsample(x)
            
            
        x = self.dec_bn3(x)
        x = self.convT3(x)
        x = self.leaky_relu(x)
        x = self.upsample(x)

        x = self.dec_bn2(x)
        x = self.convT2(x)
        x = self.leaky_relu(x)
        x = self.upsample(x)

        x = self.dec_bn1(x)
        x = self.convT1(x)
        x = self.leaky_relu(x)
        x = self.upsample(x)
        
        return x

class NaiveNetwork(nn.Module):
    def __init__(self, im_size, n_classes):
        super(NaiveNetwork, self).__init__()
        self.im_size, self.nclasses = im_size, n_classes
        
        '''self.nLayers = 3
        self.last_channel_size = 32
        
        if self.im_size == 128:
            self.nLayers = self.nLayers + 2
            self.last_channel_size = 128
            
        new_im_size = self.im_size//(2**self.nLayers)
        in_channels = new_im_size * new_im_size * self.last_channel_size
   '''
        
        # define the layers
        self.layers = nn.Sequential(
            nn.Linear(512, 256),  # z_dim = 512
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.nclasses),
            #nn.Sigmoid()
        )
        
    def forward(self,x):
        x = self.layers(x)
        return (x)

    
class DatasetMembershipClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='vgg16', num_classes=3):
        super(DatasetMembershipClassifier, self).__init__()
        
        self.hidden1 = nn.Linear(512, 512)
        self.hidden2 = nn.Linear(512, 128)
        self.fc = nn.Linear(128, num_classes)

        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, features):
        out = self.hidden1(features)
        out = self.relu(out)
        out = self.hidden2(out)
        out = self.relu(out)
        return self.fc(out)
    