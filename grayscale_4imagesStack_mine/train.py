import network 
import torch
import torch.optim as optim
import torch.nn as nn
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os


""" uses gpu to train (faster) """
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataSet = []
totalLen = 0
for arquivo in os.listdir(r'C:\Users\ferna\Desktop\bolsaMachine\Versoes carrinho\grayscale4imagesStack\pairStatesActions3'):
       with open('C:/Users/ferna/Desktop/bolsaMachine/Versoes carrinho/grayscale4imagesStack/pairStatesActions3/' + arquivo, 'rb') as file:
              data = pkl.load(file)
       dataSet.append(data)
       totalLen = totalLen + len(data)

limit = int((3/4)*(totalLen))


X = []
Y = []
for pairStateActions in dataSet:
       for i in range(len(pairStateActions)):
              X.append(pairStateActions[i][0])
              Y.append(pairStateActions[i][1])
       
X = np.array(X).astype('float32')



""" normalize data """
X = np.array([(x/255.0) for x in X]).astype('float32')


""" convert input images to grayscale """
X = np.array([np.dot(x[...,:3],[0.299,0.587,0.114]) for x in X]).astype('float32')


""" convert output actions to the desired scale, since we are using the sigmoid function and
    the binary cross entropy as loss funtion, we expect values between 0 and 1  """
def scale(y):
       for dir in y:
              dir[0] = dir[0]/2 + 0.5
       return y

""" stacks 4 frames/images to be used as input to the neural network """
def stackImages(images):
       stackedImages = []
       for i in range(len(images)):
              stackedImages.append(np.zeros((4,96,96)))
              if i > 3:
                     arrays = [images[i-3],images[i-2],images[i-1],images[i]]
                     stackedImages[i] = np.expand_dims(stackedImages[i],axis=0)
                     stackedImages[i] = np.stack(np.array(arrays),axis=0)
              else:
                     arrays = (images[i],images[i],images[i],images[i])
                     stackedImages[i] = np.expand_dims(stackedImages[i],axis=0)
                     stackedImages[i] = np.stack(np.array(arrays),axis=0)
       return stackedImages



X = np.array(stackImages(X))




""" training and validation data sets """
trainingX = np.array(X[:limit]).astype('float32')
trainingY = Y[:limit]
trainingY = np.array(scale(trainingY)).astype('float32')

validationX = np.array(X[limit:]).astype('float32')
validationY = Y[limit:]
validationY = np.array(scale(validationY)).astype('float32')


""" converts to tensor """
trainingSet = torch.utils.data.TensorDataset(torch.from_numpy(trainingX),torch.from_numpy(trainingY))



""" defines the network """
CNN = network.neuralNetwork()
CNN.to(device)

train_loader = torch.utils.data.DataLoader(trainingSet, batch_size=800)


""" training parameters """
optimizer = optim.Adam(CNN.parameters(), lr=.0001)
criterion = nn.BCELoss()
epochs = 200
limitEpochs = 3


totalLossArray = []
totalValLossArray = []
counter = 0
last_loss = 0



for epoch in range(epochs):
       total_loss_train = 0
       total_loss_val = 0
       counterVal = 0
       counterTrain = 0

       """ training """
       for batch in train_loader:
              input, target = batch

              input, target = input.to(device), target.to(device)
              predictions = CNN(input)

              loss = criterion(predictions[:,0],target[:,0]) + criterion(predictions[:,1],target[:,1]) + criterion(predictions[:,2],target[:,2])
              
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()


       """ validation """
       with torch.no_grad():
              trainInput, trainTarget = torch.from_numpy(trainingX.astype('float32')),torch.from_numpy(trainingY.astype('float32'))
              trainInput, trainTarget = trainInput.to(device), trainTarget.to(device)
              trainPredictions = CNN(trainInput)
              trainLoss = criterion(trainPredictions[:,0],trainTarget[:,0]) + criterion(trainPredictions[:,1],trainTarget[:,1]) + criterion(trainPredictions[:,2],trainTarget[:,2])
              total_loss_train += trainLoss.item()

              valInput, valTarget = torch.from_numpy(validationX.astype('float32')),torch.from_numpy(validationY.astype('float32'))
              valInput, valTarget = valInput.to(device), valTarget.to(device)
              valPredictions = CNN(valInput)
              valLoss = criterion(valPredictions[:,0],valTarget[:,0]) + criterion(valPredictions[:,1],valTarget[:,1]) + criterion(valPredictions[:,2],valTarget[:,2])
              total_loss_val += valLoss.item()


       totalLossArray.append(total_loss_train/(len(trainingX)))   
       totalValLossArray.append(total_loss_val/(len(validationX)))
       print("epoch: ", epoch, "loss: ", total_loss_train/(len(trainingX)))
       print("epoch: ", epoch, "valLoss: ",total_loss_val/(len(validationX)))



       """ early stopping """
       if total_loss_val > last_loss:
              counter += 1
              if counter >= limitEpochs:
                     break
       else:
              counter = 0
       last_loss = total_loss_val



timeStamp = str(datetime.now()).replace(" ","_")
timeStamp = timeStamp.replace(":","-")

#
torch.save(CNN.state_dict(), 'trainedCNN'+timeStamp.replace(".","")+'.pt')


with open('trainingLoss'+ timeStamp.replace(".","") +'.pkl', 'wb') as file:
       pkl.dump(totalLossArray, file) 

with open('validationLoss'+ timeStamp.replace(".","") +'.pkl', 'wb') as file:
       pkl.dump(totalValLossArray, file) 


plt.plot(totalLossArray,label="training loss")
plt.plot(totalValLossArray,label="validation loss")
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()