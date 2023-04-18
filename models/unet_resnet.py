import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from unet_resnet import SelfAttention

class UNet_resnet50enc(nn.Module):
    def __init__(self, c_in=3, c_out=3,remove_deep_conv=False):
        super().__init__()
        #self.time_dim = time_dim       
        self.remove_deep_conv = remove_deep_conv
        # New weights with accuracy 80.858%
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2) 
        # input.                        # 3, 512, 512
        self.conv1 = resnet.conv1       # 64, 256, 256
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool   # 64, 128, 128
        self.layer1 = resnet.layer1     # 256, 128, 128
        self.layer2 = resnet.layer2     # 512, 64, 64
        self.layer3 = resnet.layer3     # 1024, 32, 32
        self.layer4 = resnet.layer4     # 2048, 16, 16

        self.att = SelfAttention(2048)  # 2048, 16, 16

        self.up1 = Up(2048+1024, 1024)  # 1024, 32, 32
        self.up2 = Up(1024+512, 512)    # 512, 64, 64
        self.up3 = Up(512+256, 256)     # 256, 128, 128
        self.up4 = Up(256+64, 64)       # 64, 256, 256
        self.up5 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True) # 64, 512, 512
        self.outc = nn.Conv2d(64, 3, kernel_size=1) # 3, 512, 512

    def forward(self, x, t=None):    
        # Encoder
        x = self.conv1(x)       
        x = self.bn1(x)
        x = self.relu(x)
        x0 = self.maxpool(x)    
        x1 = self.layer1(x0)    
        x2 = self.layer2(x1)    
        x3 = self.layer3(x2)    
        x4 = self.layer4(x3)    

        att = self.att(x4)
        
        xo = self.up1(att, x3, t) 
        xo = self.up2(xo, x2, t)  
        xo = self.up3(xo, x1, t)  
        xo = self.up4(xo, x, t)  
        xo = self.up5(xo)  
        output = self.outc(xo)
        return output