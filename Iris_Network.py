# Dependencies
import csv
import tensorflow as tf
import numpy as np

# Make results reproducible
seed = 1234
np.random.seed(seed)
tf.set_random_seed(seed)

# Read .csv file
file = open('Iris_Dataset.csv', newline='')
reader = csv.reader(file)
# Skip the header ( Id,SepalLengthCm,SepalWidthCm... )
header = next(reader)

X = []
y = []

# Loading the data inside X and y
for row in reader:
    # Id,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species
        # - Iris-setosa
        # - Iris-versicolor
        # - Iris-virginica
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

# Converting X and y to two numpy arrays
X = np.array(X, dtype='float32')
y = np.array(y, dtype='float32')

# Shuffle Data
indices = np.random.choice(len(X), len(X), replace=False)
X_values = X[indices]
y_values = y[indices]

# Creating a Train and a Test Dataset
X_test = X_values[-20:]
X_train = X_values[:-20]
y_test = y_values[-20:]
y_train = y_values[:-20]

# Session
sess = tf.Session()
