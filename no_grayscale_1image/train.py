from numpy.random.mtrand import shuffle
import network 
import torch
import torch.optim as optim
import torch.nn as nn
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

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
#conjuntos de treinamento e teste
trainingX = X[:limite]
trainingY = Y[:limite]


# def scale(y):
#        for i in y:
#               if i[0] == -1:
#                      i[0] = 0
#               elif i[0] == 1:
#                      i[0] = 1
#               else:
#                      i[0] = .5
#        return y

def scale(y):
       for dir in y:
              dir[0] = dir[0]/2 + 0.5
       return y

trainingY = scale(trainingY)
#print((trainingY.shape))

testingX = X[limite:]
testingY = Y[limite:]



testingY = scale(testingY)

#converter para tensor
trainingSet = torch.utils.data.TensorDataset(torch.from_numpy(trainingX),torch.from_numpy(trainingY))
testingSet = torch.utils.data.TensorDataset(torch.from_numpy(testingX),torch.from_numpy(testingY))


CNN = network.neuralNetwork()

train_loader = torch.utils.data.DataLoader(trainingSet, batch_size=150)
test_loader = torch.utils.data.DataLoader(testingSet, batch_size=testingX.shape[0])


optimizer = optim.Adam(CNN.parameters(), lr=.00001)

criterion = nn.BCELoss()

epochs = 100
totalLossArray = []

totalVlossArray = []

for epoch in range(epochs):
       total_loss = 0
       for batch in train_loader:
              input, target = batch
              input = input.permute(0,3,1,2)

              predictions = CNN(input)

              loss = criterion(predictions[:,0],target[:,0]) + criterion(predictions[:,1],target[:,1]) + criterion(predictions[:,2],target[:,2])

              optimizer.zero_grad()
              loss.backward()
              optimizer.step()
              total_loss += loss.item()


       totalLossArray.append(total_loss)   
       print("epoch:", epoch, "loss:", total_loss)
      
torch.save(CNN.state_dict(), 'trainedCNN9.pth')


plt.plot(totalLossArray,label="training loss")
plt.plot()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()