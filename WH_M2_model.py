import torch.nn as nn
import torch
from torchvision import models
from utils import save_net,load_net
import cv2

class WHCNet(nn.Module):
    def __init__(self, load_weights=False):
        super(WHCNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        
        self.backend_feat  = [512,256,256,128,128] #,256,128,64]
        self.backend_feat2 = [128,64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat,in_channels = 512,dilation = False)
        self.backend_res = make_res_layer(512,128)
        self.backend_up_conv = make_conv(256,128,1)
        self.backend_up_conv1 = make_layers(self.backend_feat2,in_channels = 128,dilation = False)
        self.output_layer = make_output(64) 
        
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            
            for i in range(len(self.frontend.state_dict().items())):
                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]

   def forward(self,x):
        x1 = self.frontend(x)
        x2 = self.backend(x1)
        x12 = self.backend_res(x1)
        x3 = self.backend_up_conv(torch.cat([x12,x2],1))
        x4 = self.backend_up_conv1(x3)
        x = self.output_layer(x4)
        return x
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
                
def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
 
    return nn.Sequential(*layers)     
    
    
def make_res_layer(in_channel, out_channel):
    layers = []
    conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
    layers += [conv1, nn.ReLU(inplace=True)]
    return nn.Sequential(*layers) 
 
    
def make_conv(in_channel, out_channel, drate=1):
    layers = []
    conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=drate, dilation = drate)
    layers += [conv1, nn.ReLU(inplace=True)]
    return nn.Sequential(*layers) 


def make_output(in_channel):
    layers = []
    layers +=[nn.Conv2d(in_channel, 1, kernel_size=1)]
    return nn.Sequential(*layers)              
