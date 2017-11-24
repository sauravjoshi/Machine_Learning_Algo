"""
Created on Sun Sep  3 11:31:28 2017

@author: SaGa

"""

import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

'''
Our main methodology to handle with the neural networks would be as follows:-
Feedforwarding i.e passing the data straight-forward as 
Input >> Weight >> Hidden_layer_1 (Activation Function) >> weights >> Hidden_layer_2 (Activation Function) >> weights >> output_layer
Finally compare the output with the intended output done through a cost or loss function (cross_entropy)
Then use optimization function (optimizer) that is used to minimize that cost i.e.
optimization function( optimizer) >> minimize cost( AdamOptimizer)
What that does is go backwards and manipulated the weights hence a back-propagation
Also feed-forward + back-propagation = epoch
'''
# Importing the mnist data. The data would initially be downloaded and for the next iterations of use, extracted therefore.
# The one_hot attribute defines the set as follows 
# For a digit 1, set would be as [1,0,0,0,0,0,0,0,0]
# For a digit 2. set would be as [0,1,0,0,0,0,0,0,0] , Hence therefore one hot.

mnist = input_data.read_data_sets("mnist_data_cpy", one_hot=True)

# Now defining the number of nodes in the hidden layer as well as the output layer.
# Number of nodes don't need to be identical 

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

# Nodes for classes = 10 as for mnist dataset labels are 0,1,2,3,4,5,6,7,8,9 that equals 10
n_classes = 10

# Defining the batch size so as to send the data in batches to the neural  network

batch_size = 100

# Defining the placeholder for the data. These contain the initial data i.e. of shape 0 X 784 as for 28*28 sized images.
# Also the placeholder for the result set.
# x is the input data, second parameter is shape for the input_data 
# Useful to provide shape for validating the input we are feeding to the input.
# y id the placeholder for the labels 
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

# Function for our basic neural network with layers.
def neural_network_model(data):

	#Firstly creating a tensor that holds the value for all the weights and biases as a dictionary.
	hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
	'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
	'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
	'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
	'biases': tf.Variable(tf.random_normal([n_classes]))}

	# As the layer nodes are calculated as input*weights + biases , Hence peroforming a matrix multiplication for the tensors and then adding to the biases
	# using an avtivation function Rectified Linear

	l1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']), hidden_layer_3['biases'])
	l3 = tf.nn.relu(l3)			

	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

	return output

def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	hm_epochs = 10

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict={x = epoch_x, y: epoch_y})
				epoch_loss+= c
			print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print ('Accuracy', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

train_neural_network(x)
