import numpy as np


# function to read in the training data
# it creates a list of all the instances
# and a set of all the words
def read_training_data(filename):
	for line in open(filename):
		train.append(line.strip())
		instance = line.strip().split(" ")
		for j in range(len(instance)):
			word_set.add(instance[j])

# function to read in the test data
def read_test_data(filename):
	for line in open(filename):
		test.append(line.strip())

# function to transform the training and test data into numpy arrays
def fill_arrays(no_negative, no_positive, total, instance_list, array, label_array, list_of_words):
	for i in range(no_negative):
		label_array[no_positive+i] = -1

	for i in range(total):
		words_in_an_instance = instance_list[i].split(" ")
		for j in range(len(words_in_an_instance)):
			if words_in_an_instance[j] in word_list:
				array[i][list_of_words.index(words_in_an_instance[j])] = 1

# function which implements the perceptron algorithm
# at each iteration it also checks the accuracy of the weights (using the perceptron_test function)
# at the end of the alogrithm it calculates the error rates and outputs the results in a txt file
def perceptron(order, weights, b, results, filename):
	iterations = 0
	while iterations < max_iterations:
		for i in range(total_train):
			j = order[i]   
			y = train_label[j]
			a = np.sum((train_array[j] * weights)) + b
			if a*y <= 0:
				b += y
				weights += y*train_array[j]	
		results[0][iterations] = perceptron_test(total_test, weights, b, test_array, test_label)
		results[2][iterations] = perceptron_test(total_train, weights, b, train_array, train_label)
		iterations += 1

	results[1] = total_test
	results[3] = total_train
	results[4] = np.divide(results[0],results[1])
	results[5] = np.divide((results[1] - results[0]),results[1])
	results[6] = np.divide((results[3] - results[2]),results[3])

	np.savetxt(filename, results, delimiter=',')

# function to test the accuracy of the perceptron weights
def perceptron_test(instances, weights, b, array, label):
	accuracy = 0
	for i in range(instances):
		a = np.sum((weights * array[i])) + b
		y = label[i]
		if y*a > 0:
			accuracy += 1
	return accuracy


# *************************************************************************
# ******** read in data and transform it into numpy arrays ****************
# *************************************************************************

# ********************* training data *************************************

# initiate variables
train = []        
test = []
word_set = set()

# read in the positive training data file
read_training_data('train.positive')
no_positive_train= len(train)  #capture the no. of positive training instances

# read in the negative training data file
read_training_data('train.negative')
total_train= len(train)								#capture total no. of training instances
no_negative_train= total_train - no_positive_train  #capture the no. of negative training instances

#capture the no. of unique words in the training data
no_words = len(word_set)
# create a list of all words
word_list = list(word_set)

# create arrays which represent the training instances and the labels
train_array = np.zeros((total_train,no_words), dtype = np.int)
train_label = np.ones((total_train), dtype = np.int)

# transform data into numpy arrays
fill_arrays(no_negative_train, no_positive_train, total_train, train, train_array, train_label, word_list)


# ********************* test data *************************************


# read in the positive test data file
read_test_data('test.positive')
no_positive_test = len(test) #capture the no. of positive test instances

# read in the negative test data file
read_test_data('test.negative')
total_test = len(test)	#capture total no. of test instances
no_negative_test = total_test - no_positive_test #capture the no. of negative test instances

# create arrays which represent the test instances and the labels
test_array = np.zeros((total_test,no_words), dtype = np.int)
test_label = np.ones((total_test), dtype = np.int)

# transform data into numpy arrays
fill_arrays(no_negative_test, no_positive_test, total_test, test, test_array, test_label, word_list)

# *************************************************************************
# ********************** start perceptron *********************************
# *************************************************************************

# initiate variables
weights = np.zeros(no_words, dtype=np.int)
b = 0
max_iterations = 50
results = np.zeros((7,max_iterations), dtype=np.float)

# ********************* ordered data **************************************

order = np.arange(total_train) #array [0,1,2,3,4,....] 
perceptron(order, weights, b, results, 'results_ordered.txt') #run perceptron algorithm

# ********************* permuted data **************************************

# reset variables
weights = np.zeros(no_words, dtype=np.int)
b = 0
results1 = np.zeros((7,max_iterations), dtype=np.float)

order1 = np.random.choice(total_train, total_train, replace=False) # array filled with number randomly without replacement
perceptron(order1, weights, b, results1, 'results_random.txt') #run perceptron algorithm


