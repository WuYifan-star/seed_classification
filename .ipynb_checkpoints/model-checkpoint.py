
import torchvision.models as models
from torch import nn
import torchvision
import torch


def get_ResNet_net(devices):
    model = models.__dict__['resnet50']()

    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet50(pretrained=True)
    
    
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 512),
                                            nn.ReLU(),
                                            nn.Dropout(0.5),
                                            nn.Linear(512, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 2))
    finetune_net = finetune_net.to(devices[0])
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net




def get_Dino_net(devices):
    model = models.__dict__['resnet50']()

    finetune_net = nn.Sequential()
    finetune_net.features = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    
    
    finetune_net.output_new = nn.Sequential(nn.Linear(2048, 1000),
                                            nn.ReLU(),
                                            nn.Linear(1000, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 2))
    finetune_net = finetune_net.to(devices[0])
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net

