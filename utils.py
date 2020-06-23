import os
import sys
import torch.nn as nn
import torch.nn.init as init
import torch
import torchvision
import torchvision.transforms as transforms
from glob import glob

def processDataset(dataset, trainingBatchSize):
    if dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=trainingBatchSize, shuffle=True, num_workers=2)
    
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2) ##test batch size is kept fized at 100 and not varied
        return trainset, trainloader, testset, testloader
    
    if dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=trainingBatchSize, shuffle=True, num_workers=2)
    
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)  ##test batch size is kept fized at 100 and not varied
        return trainset, trainloader, testset, testloader
    
    
    if dataset == 'svhn':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
        trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=trainingBatchSize, shuffle=True, num_workers=2)
    
        testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)   ##test batch size is kept fized at 100 and not varied
        return trainset, trainloader, testset, testloader
    
def lossFunctionDefinition(): 
##defines loss function
    return nn.CrossEntropyLoss()   
    
def findCheckpointFile(checkpointPath,dataset,modelName):  
##when training is resumed from a checkpoint, finds the appropriate file for this dataset and model 
    matchingCheckpointFiles = glob(checkpointPath+dataset+modelName+'Epoch*.pth')  
    if len(matchingCheckpointFiles)==0:
        sys.exit('No matching file in checkpoint. Please run without resume option')
    elif len(matchingCheckpointFiles)==1:
        checkpointFile=matchingCheckpointFiles[0]
    elif len(matchingCheckpointFiles)>1:
        maxEpochNum=0
        for checkpointFileIter in range (0,len(matchingCheckpointFiles)):
            checkpointFileString = matchingCheckpointFiles[checkpointFileIter]
            epochNumThisFile = int(checkpointFileString[len(checkpointPath+dataset+modelName+'Epoch'):checkpointFileString.rfind('.pth')])
            maxEpochNum = epochNumThisFile if epochNumThisFile>maxEpochNum else maxEpochNum
        checkpointFile=checkpointPath+dataset+modelName+'Epoch'+str(maxEpochNum)+'.pth'
        print(checkpointFile)
 
    return checkpointFile   
    
    
def identifyCorrectSamplesWithTop5Criterion(outputs,targets):
    _, predicted = outputs.topk(5, 1, True, True)
    predicted = predicted.t()
    correct = predicted.eq(targets.view(1, -1).expand_as(predicted)).float()
    correctSamplesTop5Criterion = correct[:5].view(-1).float().sum(0, keepdim=True)
    return correctSamplesTop5Criterion