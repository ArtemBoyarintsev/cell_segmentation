import torch
import torch.nn as nn
import torchvision

class vgg16_FCN(nn.Module):
    THIRD_POOLING_INDEX = 16
    FORTH_POOLING_INDEX = 23
    def __init__(self, n_class = 1):
        super(vgg16_FCN, self).__init__()
        self.vgg16 = torchvision.models.vgg16(pretrained=True).features
        
        self.fc6 = nn.Conv2d(512, 4096, 1)
        self.relu6 = nn.ReLU(inplace=True)
        self.droput6 = nn.Dropout2d()
        
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.droput7 = nn.Dropout2d()
        
        self.direct_score = nn.Conv2d(4096, n_class, 1)
        self.third_pool = nn.Conv2d(256, n_class, 1)
        self.forth_pool = nn.Conv2d(512, n_class, 1)
        
        self.upscore2 = nn.ConvTranspose2d(n_class, n_class, 2, stride=2, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(n_class, n_class, 2, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(n_class, n_class, 8, stride=8, bias=False)
    
    def forward(self, batch):
        prev_layer_output = batch
        layers_outputs  = []
        with torch.no_grad():
            for i, layer in enumerate(self.vgg16):
                prev_layer_output = layer(prev_layer_output)
                layers_outputs.append(prev_layer_output)
           
        h = prev_layer_output
            
        h = self.fc6(h)
        h = self.relu6(h)
        h = self.droput6(h)

        h = self.fc7(h)
        h = self.relu7(h)
        h = self.droput7(h)
        
        direct_score = self.direct_score(h)
        direct_score = self.upscore2(direct_score)
        
        third_pooling_output = layers_outputs[vgg16_FCN.THIRD_POOLING_INDEX]
        forth_pooling_output = layers_outputs[vgg16_FCN.FORTH_POOLING_INDEX]
        
        forth_pool_skip_conection = self.forth_pool(forth_pooling_output)
        h = direct_score + forth_pool_skip_conection
        h = self.upscore_pool4(h)
        
        third_pool_skip_connection = self.third_pool(third_pooling_output)
        
        h = h + third_pool_skip_connection
        
        h = self.upscore8(h)
        
        return h