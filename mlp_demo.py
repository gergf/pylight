from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 
import pylight  
import numpy 

## GET DATASET ##
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_samples = mnist.train.images # Shape (55000, 784)
train_labels  = mnist.train.labels # Shape (55000, 10) one_hot_labels

test_samples = mnist.test.images # Shape (10000, 784)
test_labels  = mnist.test.labels # Shape (10000, 10) one_hot_labels

## DEFINE THE NETWORK ##
W1 = pylight.initializers.uniform().get_variable(shape=(784, 128))
W2 = pylight.initializers.uniform().get_variable(shape=(128, 10))

net_def = [
	{"type" : "dense", "W" : W1, "input_shape" : (784,), "name" : "dense:0"},
	{"type" : "sigmoid", "name" : "sigmoid:0"},
	{"type" : "dense", "W" : W2, "name" : "dense:1"},
	{"type" : "linear", "name" : "linear:0"},
	{"type" : "softmax_cross_entropy",  "name" : "cost_function:0"}
]

optimizer = pylight.optimizers.GradientDescent(learning_rate=0.5)
net = pylight.models.neural_network(net_def, optimizer)

## EVALUATE THE MODEL ##
result = net.predict(test_samples)
y_pred = numpy.argmax(result, axis=1)
y_true = numpy.argmax(test_labels, axis=1)
print("Initial accuracy", accuracy_score(y_pred, y_true))

## TRAIN THE MODEL ##
for _ in range(1000):
	x_batch, y_batch = mnist.train.next_batch(100)
	net.train(x_batch, y_batch)

## EVALUATE THE MODEL ##
result = net.predict(test_samples)
y_pred = numpy.argmax(result, axis=1)
y_true = numpy.argmax(test_labels, axis=1)
print("Post training accuracy", accuracy_score(y_pred, y_true))