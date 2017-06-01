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

# Create graph session
sess = tf.Session()

# Declare batch size
batch = 50

# Initialize placeholders ( X and y )
X_data = tf.placeholder(shape=[None, 4], dtype=tf.float32)
y_data = tf.placeholder(shape=[None, 3], dtype=tf.float32)

# Numbers of hidden neurons
hidden_neurons = 8

# Create variables for both layers
w1 = tf.Variable(tf.random_uniform(shape=[4, hidden_neurons])) # ( Weights ) Inputs -> Hidden node
b1 = tf.Variable(tf.random_uniform(shape=[hidden_neurons])) # ( First Bias )
w2 = tf.Variable(tf.random_uniform(shape=[hidden_neurons, 3])) # ( Weights ) Hidden node -> Outputs
b2 = tf.Variable(tf.random_uniform(shape=[3])) # ( Second Bias )

# Declare model operations
hidden_output = tf.nn.sigmoid(tf.add(tf.matmul(X_data, w1), b1)) # Activation Function : Sigmoid
final_output = tf.nn.softmax(tf.add(tf.matmul(hidden_output, w2), b2)) # Activation Function : Softmax

# Declare loss function ( Mean squared error )
loss = tf.reduce_mean(tf.square(y_values - final_output))

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.005)
train_step = my_opt.minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
loss_vec = []
test_loss = []
for i in range(500):
    rand_index = np.random.choice(len(X_train), size=batch)
    rand_x = X_train[rand_index]
    rand_y = np.transpose([y_train[rand_index]])
    sess.run(train_step, feed_dict={X_data: X_values, y_data: y_values})

    temp_loss = sess.run(loss, feed_dict={X_data: rand_x, y_data: rand_y})
    loss_vec.append(np.sqrt(temp_loss))

    test_temp_loss = sess.run(loss, feed_dict={X_data: x_train, y_values: np.transpose([y_test])})
    test_loss.append(np.sqrt(test_temp_loss))
    if (i+1)%50==0:
        print('Generation: ' + str(i+1) + '. Loss = ' + str(temp_loss))
