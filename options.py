"""
Code Author: Sudipan Saha.

"""

import torch
import argparse
import os
import numpy as np



class options():
    """This class defines some options/arguments
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        

    def initialize(self, parser):
        parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
        
        parser.add_argument('--dataset', default='cifar10', type=str, help='dataset')  ##dataset: cifar10/cifar100/svhn
        parser.add_argument('--lr', default='0.1', type=str, help='learning rate(s)')  ##different learning rates can be passed as comma-separated string like '0.1,0.05'
        parser.add_argument('--epochs', default='100', type=str, help='epoch count(s)')  ##epochs corr. to different learning rates can be passed as comma-separated string like '100,50'
        parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
        parser.add_argument('--checkpointPath', default='./checkpoint/', help='checkpoint path') ##checkpoint path, where trained model is stored 
        parser.add_argument('--model', default='vgg16', help='network architecture: vgg16 / vgg19 / mobilenet_v2 / resnet18 / resnet34 / resnet50 / resnet101') ##network architecture
        parser.add_argument('--sgdMomentum', default=0.9, help='momentum for SGD optimizer') ##SGD optimizer momentum
        parser.add_argument('--sgdWeightDecay', default=5e-4, help='weight decay for SGD optimizer') ##SGD optimizer weight decay
        parser.add_argument('--trainingBatchSize', default=64, type=int, help='training batch size')  ##batch size during training
        self.initialized = True
        return parser

    

    def parseOptions(self):
        """Parse the options"""
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)
        opt = parser.parse_args()
        
        dataset = opt.dataset ##valid values:  cifar10/cifar100/svhn/mnist

        checkpointPath = opt.checkpointPath ## checkpoint path
        if not os.path.exists(checkpointPath):
            os.makedirs(checkpointPath)       ##Creating checkpoint directory if it does not exist
            
        resumeTrainingBool = opt.resume ##this bool indicates whether training is to be resumed from a checkpoint
        
        learningRates = np.array(opt.lr.split(','),dtype=np.float).tolist()  ##learning rates, can be one value or an array
        
        epochs = np.array(opt.epochs.split(','),dtype=np.int).tolist() ##epochs, can be one value or an array
        
        modelName = opt.model  ##network architecture 
        
        sgdMomentum = opt.sgdMomentum ##SGD optimizer momentum
        
        sgdWeightDecay = opt.sgdWeightDecay  ##SGD optimizer weight decay
        
        trainingBatchSize = opt.trainingBatchSize ##Batch size during training
        
        return dataset,checkpointPath,resumeTrainingBool,learningRates,epochs, modelName, sgdMomentum, sgdWeightDecay, trainingBatchSize
   