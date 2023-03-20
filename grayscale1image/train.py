from numpy.random.mtrand import shuffle
import network 
import torch
import torch.optim as optim
import torch.nn as nn
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

with open('pairStatesActions.pkl', 'rb') as file:
       dataSet = pkl.load(file)

with open('pairStatesActions2.pkl', 'rb') as file2:
       dataSet2 = pkl.load(file2)

with open('pairStatesActions3.pkl', 'rb') as file3:
       dataSet3 = pkl.load(file3)

limite = int((3/4)*(len(dataSet) + len(dataSet2) + len(dataSet3)))

X1 = np.array([dataSet[i][0] for i in range(len(dataSet))]).astype('float32')
Y1 = np.array([dataSet[i][1] for i in range(len(dataSet))]).astype('float32')

X2 = np.array([dataSet2[i][0] for i in range(len(dataSet2))]).astype('float32')
Y2 = np.array([dataSet2[i][1] for i in range(len(dataSet2))]).astype('float32')

X3 = np.array([dataSet3[i][0] for i in range(len(dataSet3))]).astype('float32')
Y3 = np.array([dataSet3[i][1] for i in range(len(dataSet3))]).astype('float32')

X = np.append(X1,X2,axis=0)
X = np.append(X,X3,axis=0)

Y = np.append(Y1,Y2,axis=0)
Y = np.append(Y,Y3,axis=0)


#normalizar
X = np.array([(x/128.0)-1 for x in X]).astype('float32')

#grayscale
X = np.array([np.dot(x[...,:3],[0.299,0.587,0.114]) for x in X]).astype('float32')

def scale(y):
       for dir in y:
              dir[0] = dir[0]/2 + 0.5
       return y

#conjuntos de treinamento e teste
trainingX = X[:limite]
trainingY = Y[:limite]
trainingY = scale(trainingY)


validatingX = X[limite:]
validatingY = Y[limite:]
validatingY = scale(validatingY)

#converter para tensor
trainingSet = torch.utils.data.TensorDataset(torch.from_numpy(trainingX),torch.from_numpy(trainingY))
#validatingSet = torch.utils.data.TensorDataset(torch.from_numpy(validatingX),torch.from_numpy(validatingY))


CNN = network.neuralNetwork()

train_loader = torch.utils.data.DataLoader(trainingSet, batch_size=200)
#test_loader = torch.utils.data.DataLoader(validatingSet, batch_size=validatingX.shape[0])


optimizer = optim.Adam(CNN.parameters(), lr=.0001)

criterion = nn.BCELoss()

epochs = 100
totalLossArray = []

totalValLossArray = []

for epoch in range(epochs):
       total_loss_train = 0
       total_loss_val = 0

       for batch in train_loader:
              input, target = batch
              input = input.unsqueeze(0)
              input = input.permute(1,0,2,3)
              
              predictions = CNN(input)

              loss = criterion(predictions[:,0],target[:,0]) + criterion(predictions[:,1],target[:,1]) + criterion(predictions[:,2],target[:,2])

              optimizer.zero_grad()
              loss.backward()
              optimizer.step()


       with torch.no_grad():
              trainInput, trainTarget = torch.from_numpy(trainingX.astype('float32')),torch.from_numpy(trainingY.astype('float32'))
              trainInput = trainInput.unsqueeze(0)
              trainInput = trainInput.permute(1,0,2,3)
              trainPredictions = CNN(trainInput)
              trainLoss = criterion(trainPredictions[:,0],trainTarget[:,0]) + criterion(trainPredictions[:,1],trainTarget[:,1]) + criterion(trainPredictions[:,2],trainTarget[:,2])
              total_loss_train += trainLoss.item()


              valInput, valTarget = torch.from_numpy(validatingX.astype('float32')),torch.from_numpy(validatingY.astype('float32'))
              valInput = valInput.unsqueeze(0)
              valInput = valInput.permute(1,0,2,3)
              valPredictions = CNN(valInput)
              valLoss = criterion(valPredictions[:,0],valTarget[:,0]) + criterion(valPredictions[:,1],valTarget[:,1]) + criterion(valPredictions[:,2],valTarget[:,2])
              total_loss_val += valLoss.item()

       totalLossArray.append(total_loss_train/(len(trainingX)))   
       totalValLossArray.append(total_loss_val/(len(validatingX)))
       print("epoch:", epoch, "trainLoss:", total_loss_train/len(trainingX), "valLoss:",total_loss_val/len(validatingX))
      
torch.save(CNN.state_dict(), 'trainedCNN.pth')

timeStamp = str(datetime.now()).replace(" ","_")
timeStamp = timeStamp.replace(":","-")

with open('trainingLoss_umaimagem'+ timeStamp.replace(".","") +'.pkl', 'wb') as file:
       pkl.dump(totalLossArray, file) 

with open('validatingLoss_umaimagem'+ timeStamp.replace(".","") +'.pkl', 'wb') as file:
       pkl.dump(totalValLossArray, file) 




plt.plot(totalLossArray,label="training loss")
plt.plot(totalValLossArray,label="validation loss")
plt.legend()
plt.plot()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()