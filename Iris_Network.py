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

# Batch size
batch_size = 50

# Initialize placeholders
X_data = tf.placeholder(shape=[None, 4], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 3], dtype=tf.float32)

# Hidden neurons
hidden_layer_nodes = 8

# Create variables for Neural Network layers
w1 = tf.Variable(tf.random_normal(shape=[4,hidden_layer_nodes])) # Inputs -> Hidden Layer
b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))   # First Bias
w2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes,3])) # Hidden layer -> Outputs
b2 = tf.Variable(tf.random_normal(shape=[3]))   # Second Bias

# Declare model operations
hidden_output = tf.nn.sigmoid(tf.add(tf.matmul(X_data, w1), b1))
final_output = tf.nn.softmax(tf.add(tf.matmul(hidden_output, w2), b2))

# Cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(y_target * tf.log(final_output), axis=0))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Training
epoch = 500
for i in range(1, (epoch + 1)):
    sess.run(optimizer, feed_dict={X_data: X_train, y_target: y_train})
    if i % 50 == 0:
        print('Epoch', i, '|', 'Loss:', sess.run(cost, feed_dict={X_data: X_train, y_target: y_train}))
