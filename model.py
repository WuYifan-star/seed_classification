#!/usr/bin/env python
# coding: utf-8

# In[2]:


from d2l import torch as d2l
from torch import nn
from torch.nn import functional as F
import numpy as np
import torch
from torchvision import models
import torchvision
import math


# In[2]:


def get_ResNet_net(devices):
    model = models.__dict__['resnet50']()

    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet50(pretrained=True)
    
    
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 512),
                                            nn.ReLU(),
                                            nn.Dropout(0.5),
                                            nn.Linear(512, 256),
                                            nn.ReLU(),
                                            nn.Dropout(0.5),
                                            nn.Linear(256, 2))
    finetune_net = finetune_net.to(devices[0])
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net


# In[3]:


def get_Dino_net(devices):
    model = models.__dict__['resnet50']()

    finetune_net = nn.Sequential()
    finetune_net.features = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    
    
    finetune_net.output_new = nn.Sequential(nn.Linear(2048, 1000),
                                            nn.ReLU(),
                                            nn.Dropout(0.5),
                                            nn.Linear(1000, 512),
                                            nn.ReLU(),
                                            nn.Dropout(0.5),
                                            nn.Linear(512, 256),
                                            nn.ReLU(),
                                            nn.Dropout(0.5),
                                            nn.Linear(256, 2))
    finetune_net = finetune_net.to(devices[0])
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net


# In[4]:


def get_Dino_Vit_net(devices):
    linear_keyword = 'head'
    finetune_net = nn.Sequential()
    finetune_net.features = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')

    finetune_net.output_new = nn.Sequential(nn.Linear(768, 256),
                                            nn.ReLU(),
                                            nn.Dropout(0.5),
                                            nn.Linear(256, 128),
                                            nn.ReLU(),
                                            nn.Dropout(0.5),
                                            nn.Linear(128, 2))
    finetune_net = finetune_net.to(devices[0])
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    
    return finetune_net


# In[ ]:


def get_Vit_net(devices):
    linear_keyword = 'head'
    finetune_net = nn.Sequential()
    finetune_net.features = torch.hub.load('facebookresearch/deit:main','deit_tiny_patch16_224', pretrained=True)

    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 512),
                                            nn.ReLU(),
                                            nn.Dropout(0.5),
                                            nn.Linear(512, 256),
                                            nn.ReLU(),
                                            nn.Dropout(0.5),
                                            nn.Linear(256, 128),
                                            nn.ReLU(),
                                            nn.Dropout(0.5),
                                            nn.Linear(128, 2))
    finetune_net = finetune_net.to(devices[0])
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    
    return finetune_net

