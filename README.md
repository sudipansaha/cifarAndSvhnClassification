# Cifar10/100 and SVHN classification using PyTorch

An example usage: <br/>
python main.py --dataset cifar100 --trainingBatchSize 128 --lr 0.1,0.05,0.01 --epochs 2,1,1 --model resnet18
<br/>
Observe that training can be performed with different learning rates. They are passed as comma separated values with argument lr.
Epochs corresponding to each lr are also passed as comma separated values with argument epochs.
<br/>
To resume training from checkpoint, use --resume
<br/>
Detailed options (argument lists) are explained in options.py
<br/>

Codes are segregated in following files: <br/>
**main.py**: main code, training and testing <br/>
**options.py**: Argument parsing <br/>
**defineModelArchitecture.py**: defines the network architecture (vgg16/vgg19/resnet18/resnet34/resnet50/resnet101/mobilenet_v2) <br/>
**utils.py**: Some useful functions, e.g., downloading and processing dataset, definition of loss function, fetching
checkpoint file (in case training is resumed) 
<br/>
