import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np

class Network(torch.nn.Module):

    def __init__(self):
        super(Network,self).__init__()
        self.dense_input = nn.Linear(4,16)
        self.dense_hidde = nn.Linear(16,7)
        self.dense_outpt = nn.Linear(7,3)

    def forward(self,x):
        x =  F.relu(self.dense_input(x))
        x =  F.relu(self.dense_hidde(x))
        x =  self.dense_outpt(x)  #F.log_softmax si può non mettere se come cost function si usa CrossEntropyLoss.
        return x

def loadData():
    path = 'Iris_Dataset.csv'
    file = open(path, newline='')
    reader = csv.reader(file)

    # Skip the header ( Id,SepalLengthCm,SepalWidthCm... )
    header = next(reader)

    X = []
    y = []

    for row in reader:
        X.append([float(row[1]), float(row[2]), float(row[3]), float(row[4])])

        if row[5] == 'Iris-setosa':
            y.append([0])
        elif row[5] == 'Iris-versicolor':
            y.append([1])
        elif row[5] == 'Iris-virginica':
            y.append([2])


    # Converting X and y to 2 Torch Tensors
    X = torch.Tensor(X)
    y = torch.LongTensor(y)

    X = Variable(X.view(150,1,4))
    Y = Variable(y)
    return X,Y



X,Y = loadData()
print(X[0],Y[0])
net = Network()
#Training:
crit  = nn.CrossEntropyLoss()            #Softmax più CrossEntropyLoss
optim = torch.optim.SGD(net.parameters(),lr=.01)

for i in range(200):
    avg_loss = 0
    for t in range(len(Y)):
        x = X[t]
        p = net(x)
        loss = crit(p,Y[t])
        optim.zero_grad()
        loss.backward()
        optim.step()
        avg_loss += loss.data[0]
    print("Average Loss:{}".format(avg_loss/len(X)))

print("Real:{} Net:{}".format(Y[17],net(X[17])[0]))
