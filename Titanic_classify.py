import pandas as pd 
import numpy as np 
from math import floor
import tensorflow as tf

train_file = pd.read_csv('/Users/nishanthta/Downloads/train.csv')
test_file = pd.read_csv('/Users/nishanthta/Downloads/test.csv')

#hyperparameters
learning_rate = 0.2
layerdims = [6,3,4,1]
epoch = 8000

#defining inputs and outputs
X = tf.placeholder(tf.float32,[6,891])
Y = tf.placeholder(tf.float32,[1,891])
	
#weights
W1 = tf.get_variable("W1", shape=[layerdims[1],layerdims[0]], initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable("W2", shape=[layerdims[2],layerdims[1]], initializer=tf.contrib.layers.xavier_initializer())
W3 = tf.get_variable("W3", shape=[layerdims[3],layerdims[2]], initializer=tf.contrib.layers.xavier_initializer())

#biases
b1 = tf.Variable(tf.zeros([layerdims[1],1]))
b2 = tf.Variable(tf.zeros([layerdims[2],1]))
b3 = tf.Variable(tf.zeros([layerdims[3],1]))

#Layer outputs
L1 = tf.nn.tanh(tf.matmul(W1,X) + b1)
L2 = tf.nn.tanh(tf.matmul(W2,L1) + b2)
L3 = tf.sigmoid(tf.matmul(W3,L2) + b3)

#Compute cost and optimize
cost = tf.reduce_mean(-Y*tf.log(L3) - (1 - Y)*tf.log(1 - L3))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#initialize variables
init = tf.global_variables_initializer()

#get input and normalize
xdata = np.zeros((6,891))
ydata = np.zeros((1,891))
max_pclass = 3
for i in range(891):
	if np.isnan(train_file['Pclass'][i]):
		xdata[0][i] = 0.5
	else:
		xdata[0][i] = train_file['Pclass'][i] / max_pclass

for i in range(891):
	if train_file['Sex'][i] == 'male':
			xdata[1][i] = 0
	elif train_file['Sex'][i] == 'female':
			xdata[1][i] = 1

max_age = 100.0
for i in range(891):
	if np.isnan(train_file['Age'][i]):
		xdata[2][i] = 0.5
	else:
		xdata[2][i] = train_file['Age'][i] / max_age

max_s = 10.0
for i in range(891):
	if np.isnan(train_file['SibSp'][i]):
		xdata[3][i] = 0.5
	else:
		xdata[3][i] = train_file['SibSp'][i] / max_s

max_p = 10.0
for i in range(891):
	xdata[4][i] = train_file['Parch'][i] / max_p

for i in range(891):
	if train_file['Embarked'][i] == 'C':
		xdata[5][i] = 0
	elif train_file['Embarked'][i] == 'Q':
		xdata[5][i] = 0.5	
	else:
		xdata[5][i] = 1

#get output
for i in range(891):
	ydata[0][i] = train_file['Survived'][i]

#process test file
xtest = np.zeros((6,418))
ytest = np.zeros((1,418))
max_pclass = 3
for i in range(418):
	if np.isnan(test_file['Pclass'][i]):
		xtest[0][i] = 0.5
	else:
		xtest[0][i] = test_file['Pclass'][i] / max_pclass

for i in range(418):
	if test_file['Sex'][i] == 'male':
			xtest[1][i] = 0
	elif test_file['Sex'][i] == 'female':
			xtest[1][i] = 1

max_age = 100.0
for i in range(418):
	if np.isnan(test_file['Age'][i]):
		xtest[2][i] = 0.5
	else:
		xtest[2][i] = test_file['Age'][i] / max_age

max_s = 10.0
for i in range(418):
	if np.isnan(test_file['SibSp'][i]):
		xtest[3][i] = 0.5
	else:
		xtest[3][i] = test_file['SibSp'][i] / max_s

max_p = 10.0
for i in range(418):
	xtest[4][i] = test_file['Parch'][i] / max_p

for i in range(418):
	if test_file['Embarked'][i] == 'C':
		xtest[5][i] = 0
	elif test_file['Embarked'][i] == 'Q':
		xtest[5][i] = 0.5	
	else:
		xtest[5][i] = 1

#train
with tf.Session() as sess:
	sess.run(init)

	for cnt in range(epoch):
		sess.run(optimizer,feed_dict = {X: xdata,Y: ydata})
		if cnt%500 == 0:
			print(sess.run(cost,feed_dict = {X: xdata,Y: ydata}))

	print('cost after training ', sess.run(cost,feed_dict = {X: xdata,Y: ydata}))
	answer = tf.equal(tf.floor(L3 + 0.5),Y)
	accuracy = tf.reduce_mean(tf.cast(answer,"float"))

	print('train accuracy is ', sess.run(accuracy,feed_dict = {X: xdata,Y: ydata})*100, '%')

	#test
	X1 = tf.placeholder(tf.float32,[6,418])	
	LO1 = tf.nn.tanh(tf.matmul(W1,X1) + b1)
	LO2 = tf.nn.tanh(tf.matmul(W2,LO1) + b2)
	LO3 = tf.sigmoid(tf.matmul(W3,LO2) + b3)
	ytest = sess.run(LO3,feed_dict = {X1 : xtest})
	
	for i in range(418):
		print(floor(ytest[0][i] + 0.5))



			
