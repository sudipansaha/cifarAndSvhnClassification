'''Train CIFAR10 / CIFAR100 / SVHN with PyTorch.

Code Author: Sudipan Saha.

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import os
import sys
import argparse
import numpy as np

from options import options
from utils import processDataset,lossFunctionDefinition, findCheckpointFile, identifyCorrectSamplesWithTop5Criterion
from defineModelArchitecture import defineModelArchitecture




### Training function
def train(epoch):
     
    net.train() ##setting model to Train mode
    trainLoss = 0
    correctSamples = 0
    totalSamples = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()  ##setting gradients of all model parameters to zero
        outputs = net(inputs)  ##prodicting output
#        print(inputs.shape)
#        print(outputs.shape)
#        x=10/0
        loss = lossFunction(outputs, targets)  ##loss for this batch
        loss.backward()
        optimizer.step()

        trainLoss += loss.item() ##training loss for this epoch
        _, predicted = outputs.max(1)
        totalSamples += targets.size(0)
        correctSamples += predicted.eq(targets).sum().item()  ##number of correct samples in this epoch 

        print('\nEpoch: %d' % epoch, batch_idx+1,'/',len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (trainLoss/(batch_idx+1), 100.*correctSamples/totalSamples, correctSamples, totalSamples))

### Test function
def test(epoch,checkpointPath,dataset,modelName):

    net.eval()  ##setting model to eval model
    testLoss = 0
    correctSamples = 0
    correctSamplesTop5 = 0
    totalSamples = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            totalSamples += targets.size(0)
            correctSamples += predicted.eq(targets).sum().item()
            print(batch_idx+1,'/',len(testloader), 'Acc: %.3f%% (%d/%d)'% (100*correctSamples/totalSamples, correctSamples, totalSamples))
            correctSamplesTop5 += identifyCorrectSamplesWithTop5Criterion(outputs,targets)
            
    print('Test accuracy is %.2f%%'%(100*correctSamples/totalSamples))
    print('Test accuracy (Top-5) is %.2f%%'%(100*correctSamplesTop5/totalSamples))
    # Saving checkpoint
    acc = 100.*correctSamples/totalSamples
    print('Saving model in checkpoint directory')
    
    stateDict = {
       'net': net.state_dict(),
       'acc': acc,
       'epoch': epoch,
    }
        
        
    torch.save(stateDict, checkpointPath+dataset+modelName+'Epoch'+str(epoch)+'.pth')


                     
                     
if __name__ == '__main__':

    ##Parsing options
    dataset,checkpointPath,resumeTrainingBool,learningRates,epochs, modelName, sgdMomentum, sgdWeightDecay, trainingBatchSize  = options().parseOptions()
    
    ##Check whether learning rate array and epoch array has same dimension
    if len(learningRates) != len(epochs):
        sys.exit('Learning rate array and epoch array must have same dimension')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  ##defining device
    net = defineModelArchitecture().importModel(dataset,modelName)  ##importing the network 
    net = net.to(device)  ##copying network to device   
    
    if device == 'cuda':
        net = torch.nn.DataParallel(net) 
        cudnn.benchmark = True
        
    currentEpochNumber = 0  ##current epoch number starts from 0 unless training is resumed from checkpoint
    if resumeTrainingBool:  # checking whether to resume from checkpoint
        print('Resume training from checkpoint')
        checkpointFile = findCheckpointFile(checkpointPath,dataset,modelName)
        checkpoint = torch.load(checkpointFile)
        net.load_state_dict(checkpoint['net'])  ##copying network from checkpoint
        currentEpochNumber = checkpoint['epoch']  ##current epoch number is set to that from checkpoint

    trainset, trainloader, testset, testloader = processDataset(dataset, trainingBatchSize) ##processing dataset
    lossFunction = lossFunctionDefinition()  ##defining loss function

    ##multiple learning rates are supported and they are stored in the variable 'learningRates'
    ##We iterate over each learningRate in learningRates and train for "epoch" corresponding to
    ##that learningRate
    for iterationIndex,learningRate in enumerate(learningRates):  
        correspondingEpochs = epochs[iterationIndex]
        optimizer = optim.SGD(net.parameters(), lr=learningRate,momentum=sgdMomentum, weight_decay=sgdWeightDecay) ##We define SGD optimizer, however can be changed to Adam
        for epochIter in range(correspondingEpochs):  ##Iterating over the epoch
            currentEpochNumber = currentEpochNumber+1  ##updating current epoch number
            train(currentEpochNumber)                  ##Performing training for current epoch number
    test(currentEpochNumber,checkpointPath,dataset,modelName)