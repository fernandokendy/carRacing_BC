import torch
import torch.nn as nn
import torch.nn.functional as F

class neuralNetwork(nn.Module):
    def __init__(self,entrada=4,saida=3):
        super(neuralNetwork,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=entrada,out_channels=32,kernel_size=8,stride=4)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=4,stride=2)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=32,kernel_size=3,stride=1)
        self.fc1 = nn.Linear(1568,512)
        self.fcOut = nn.Linear(512,saida)
        self.dropout = nn.Dropout(0.2)


    
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.contiguous().view(x.size(0),-1) 
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fcOut(x)
        
        x[:,0] = torch.tanh(x[:,0])
        x[:,1] = torch.sigmoid(x[:,1])
        x[:,2] = torch.sigmoid(x[:,2])
        return x