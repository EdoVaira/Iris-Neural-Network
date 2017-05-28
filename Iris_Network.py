# Dependencies
import csv
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
np.random.seed(123)
dtype = torch.FloatTensor

# Open the CSV File
path = 'Iris_Dataset'
file = open(path, newline='')
reader = csv.reader(file)

# Skip the header ( Id,SepalLengthCm,SepalWidthCm... )
header = next(reader)

X = []
y = []

# Load the data in X and y
    # Id,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species
    # Iris-setosa
    # Iris-versicolor
    # Iris-virginica
for row in reader:

    Id = int(row[0])
    SepalLengthCm = float(row[1])
    SepalWidthCm = float(row[2])
    PetalLengthCm = float(row[3])
    PetalWidthCm = float(row[4])
    X.append([SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm])

    # One Hot Encoding
    if row[5] == 'Iris-setosa':
        y.append([0,0,1])
    elif row[5] == 'Iris-versicolor':
        y.append([0,1,0])
    elif row[5] == 'Iris-virginica':
        y.append([1,0,0])

# Converting X and y to 2 Torch Tensors
X = torch.Tensor(X)
y = torch.Tensor(y)

# Neural Network

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 4, 10, 3

# Create Tensors to hold input (X) and outputs (y), and wrap them in Variables.
# Setting requires_grad=False indicates that we do not need to compute gradients
# with respect to these Variables during the backward pass.
X = Variable(X.type(dtype), requires_grad=False)
y = Variable(y.type(dtype), requires_grad=False)

# Create random Tensors for weights, and wrap them in Variables.
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Variables during the backward pass.
w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

learning_rate = 1e-6
for t in range(500):
  # Forward pass: compute predicted y using operations on Variables; these
  # are exactly the same operations we used to compute the forward pass using
  # Tensors, but we do not need to keep references to intermediate values since
  # we are not implementing the backward pass by hand.
  y_pred = X.mm(w1).clamp(min=0).mm(w2)
  F.log_softmax(y_pred)

  # Compute and print loss using operations on Variables.
  # Now loss is a Variable of shape (1,) and loss.data is a Tensor of shape
  # (1,); loss.data[0] is a scalar value holding the loss.
  loss = (y_pred - y).pow(2).sum()
  print(t, '|', 'Loss :', loss.data[0])

  # Use autograd to compute the backward pass. This call will compute the
  # gradient of loss with respect to all Variables with requires_grad=True.
  # After this call w1.grad and w2.grad will be Variables holding the gradient
  # of the loss with respect to w1 and w2 respectively.
  loss.backward()

  # Update weights using gradient descent; w1.data and w2.data are Tensors,
  # w1.grad and w2.grad are Variables and w1.grad.data and w2.grad.data are
  # Tensors.
  w1.data -= learning_rate * w1.grad.data
  w2.data -= learning_rate * w2.grad.data

  # Manually zero the gradients
  w1.grad.data.zero_()
  w2.grad.data.zero_()


# new = [[New Data Here]]
# new = torch.Tensor(new)
# new = Variable(new)
# print(new.mm(w1).mm(w2))
