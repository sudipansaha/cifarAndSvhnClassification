"""
Code Author: Sudipan Saha.

"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np



class defineModelArchitecture():
    """This class imports the model architecture
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        

    def getNumOutputClasses(self, dataset):
        if dataset == 'cifar10':
            numOutputClasses = 10
        elif dataset == 'cifar100':
            numOutputClasses = 100
        elif dataset == 'svhn':
            numOutputClasses = 10
        return numOutputClasses

    

    def importModel(self, dataset, modelName):
        numOutputClasses = self.getNumOutputClasses (dataset)
        net = getattr(torchvision.models, modelName)()
        if modelName == 'vgg16':
            net.classifier._modules['6'] =  nn.Linear(4096,numOutputClasses)
        elif modelName == 'vgg19':
            net.classifier._modules['6'] =  nn.Linear(4096,numOutputClasses)
        elif modelName == 'mobilenet_v2':
            net.classifier._modules['1'] =  nn.Linear(1280,numOutputClasses)
        elif modelName == 'resnet18':
            net.fc =  nn.Linear(512,numOutputClasses)
        elif modelName == 'resnet34':
            net.fc =  nn.Linear(512,numOutputClasses)
        elif modelName == 'resnet50':
            net.fc =  nn.Linear(2048,numOutputClasses)
        elif modelName == 'resnet101':
            net.fc =  nn.Linear(2048,numOutputClasses)
        return net
   