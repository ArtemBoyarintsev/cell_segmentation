import torch
import torch.nn as nn
import torchvision

class UNET(nn.Module):
    THIRD_POOLING_INDEX = 16
    FORTH_POOLING_INDEX = 23
    def __init__(self, n_class = 1):
        super(UNET, self).__init__()
        
        # Contracting Path
        self.c1 = UNET.get_conv2d_block(3, 16, 3, 1)
        self.p1 = nn.MaxPool2d(2)
        self.d1 = nn.Dropout2d()

        self.c2 = UNET.get_conv2d_block(16, 32, 3, 1)
        self.p2 = nn.MaxPool2d(2)
        self.d2 = nn.Dropout2d()
        
        self.c3 = UNET.get_conv2d_block(32, 64, 3, 1)
        self.p3 = nn.MaxPool2d(2)
        self.d3 = nn.Dropout2d()

        
        self.c4 = UNET.get_conv2d_block(64, 128, 3, 1)
        self.p4 = nn.MaxPool2d(2)
        self.d4 = nn.Dropout2d()

        self.c5 = UNET.get_conv2d_block(128, 256, 3, 1)
        
        self.u6 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)
        self.d6 = nn.Dropout2d()
        self.c6 = UNET.get_conv2d_block(256, 128, 3, 1)
        
        self.u7 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        self.d7 = nn.Dropout2d()
        self.c7 = UNET.get_conv2d_block(128, 64, 3, 1)
        
        self.u8 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0)
        self.d8 = nn.Dropout2d()
        self.c8 = UNET.get_conv2d_block(64, 32, 3, 1)
        
        self.u9 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0)
        self.d9 = nn.Dropout2d()
        self.c9 = UNET.get_conv2d_block(32, 16, 3, 1)

        self.c10 = nn.Conv2d(16, 1, 1)
        self.activation = nn.Sigmoid()
        
        #outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    def forward(self, batch):
        c1_output = self.c1(batch)
        h = c1_output
        h = self.p1(h)
        h = self.d1(h)
        
        c2_output = self.c2(h)
        h = c2_output
        h = self.p2(h)
        h = self.d2(h)
        
        c3_output = self.c3(h)
        h = c3_output
        h = self.p3(h)
        h = self.d3(h)
        
        c4_output = self.c4(h)
        h = c4_output
        h = self.p4(h)
        h = self.d4(h)
        
        h = self.c5(h)
        
        u = self.u6(h)
        h = torch.cat((u, c4_output), dim=(1))
        h = self.d6(h)
        h = self.c6(h)
        
        u = self.u7(h)
        h = torch.cat((u, c3_output), dim=(1))
        h = self.d7(h)
        h = self.c7(h)
        
        u = self.u8(h)
        h = torch.cat((u, c2_output), dim=(1))
        h = self.d8(h)
        h = self.c8(h)
        
        u = self.u9(h)
        h = torch.cat((u, c1_output), dim=(1))
        h = self.d9(h)
        h = self.c9(h)
        
        h = self.c10(h)
        ret = self.activation(h)
        return ret 
    
    
    @staticmethod
    def get_conv2d_block(input_size, output_size, kernel_size, padding):
        """Function to add 2 convolutional layers with the parameters passed to it"""
        # first layer
        # kernel_initializer = 'he_normal', padding = 'same'
        conv2d_block = nn.Sequential()
        conv2d = nn.Conv2d(input_size, output_size, kernel_size = kernel_size, padding=padding)
        
        conv2d_block.add_module('conv_0', conv2d)
        conv2d_block.add_module('batchnorm_0', nn.BatchNorm2d(output_size))
        conv2d_block.add_module('relu0', nn.ReLU())
        
        conv2d_2 = nn.Conv2d(output_size, output_size, kernel_size=kernel_size, padding=padding)
        conv2d_block.add_module('conv_1', conv2d_2)
        conv2d_block.add_module('batchnorm_1', nn.BatchNorm2d(output_size))
        conv2d_block.add_module('relu0', nn.ReLU())

        return conv2d_block