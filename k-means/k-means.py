import numpy as np

# function to read in the training data
# it creates a list of all the instances
# and a set of all the words
def read_data(filename):
	for line in open(filename):
		data.append(line.strip())
		instance = line.strip().split(" ")
		for j in range(1,len(instance)):
			word_set.add(instance[j])

# function to transform the data into numpy arrays
def fill_arrays(total, instance_list, array, label_array_char, list_of_words, label_array_float):
	for i in range(total):
		words_in_an_instance = instance_list[i].split(" ")
		label_array_char[i] = words_in_an_instance[0]
		for j in range(1,len(words_in_an_instance)):
			if words_in_an_instance[j] in list_of_words:
				array[i][list_of_words.index(words_in_an_instance[j])] = 1

# function to convert the label array from strings to floats
def convert_label_array(label_array_char, categories, label_array_float, total):
	for i in range(len(categories)):
		locations = np.where(data_label_char == categories[i])[0]
		for j in locations:
			label_array_float[j] = i

# function to calculate the euclidean distance between 2 vectors
def euclidean_distance(x,y):
	distance = np.sqrt(np.sum(np.square(np.subtract(y,x))))
	return distance

# function to normalize a matrix of vectors
def normalize_array(x):
	for i in range(0,x.shape[0]):
		norm = np.sqrt(np.sum(x[i,:] * x[i,:]))
		x[i,:] = x[i,:] / norm
	return x	

# function to perform the k-means algorithm 
# also calculates the accuracy metrics of the output
def k_means(k, version):
	# initialize random instances as the means	
	random = np.random.choice(total_data_instances, k, replace=False)
	means_array = np.zeros((k, no_words))

	for i in range(k):
		means_array[i] = np.copy(data_array[random[i]])	
		
	# initialize variables
	distances = np.zeros(k, dtype = np.float)
	closest_mean = np.zeros(total_data_instances, dtype = np.float)
	total_label = np.zeros(k)
	converge = 0
	iterations = 0

	# repeat until converged
	while converge == 0:
		# reset variables 
		previous_labels = np.copy(closest_mean)

		# for each instance calculate the distance from each mean
		# find the closest mean
		for i in range(total_data_instances):
			for j in range(k):
				distances[j] = euclidean_distance(means_array[j], data_array[i])
			closest_mean[i] = np.argmin(distances)

		# check for convergence
		if np.array_equal(closest_mean,previous_labels) == True:
			converge = 1 #ends the while loop


		# if not converged
		# calcluate new means
		else:
			# first - reset variables 
			means_array = np.zeros((k, len(word_list)))
			total_label = np.zeros(k)
			temp = np.copy(means_array)
			
			for i in range(k):
				for j in range(total_data_instances):
					if closest_mean[j] == i:
						means_array[i] += data_array[j]
						total_label[i] += 1
			for i in range(k):
				if total_label[i] != 0:
					means_array[i] = np.divide(means_array[i], total_label[i])
				else: 
					means_array[i] = temp[i]
			

			# for Q5 of assignment
			# select the instance that is closest to the mean as the cluster centre
			if version == "closest_to_mean":
				temp_array = np.copy(means_array)
				means_array = np.zeros((k,no_words), dtype= np.float)
				distances2 = np.zeros(total_data_instances, dtype = np.float)
				closest_instance = np.zeros(k, dtype=float)
				for j in range(k):
					for i in range(total_data_instances):
						distances2[i] = euclidean_distance(temp_array[j], data_array[i])
					closest_instance[j] = np.argmin(distances2)
					means_array[j] = data_array[closest_instance[j]]

			iterations += 1


	# ************** calculating accuracy ***********************

	# initialise variables
	count_cluster_cat = np.zeros((k,8), dtype = np.int)
	k_classes = np.zeros(k, dtype = np.int)
	rows_deleted = []

	# find the most common label in each cluster
	for i in range(k):
		locations = np.where(closest_mean == i)[0]
		if len(locations) != 0:
			cluster_cat = np.zeros(len(locations), dtype = np.int)
			for j in range(len(locations)):
				cluster_cat[j] = data_label_float[locations[j]]
			count_cluster_cat[i] = np.histogram(cluster_cat, bins = np.arange(9))[0]
			k_classes = np.argmax(count_cluster_cat, axis=1)
		
		# if a cluster has no instances it will be deleted 
		else:
			rows_deleted.append(i)

	rows_deleted.sort()
	rows_deleted = rows_deleted[::-1]
	
	for i in rows_deleted:
		count_cluster_cat = np.delete(count_cluster_cat, i, axis=0)	
		k_classes = np.delete(k_classes, i, axis=0)

	# merge clusters with the same label
	rows_deleted = []
	for i in range(8):
		duplicates = np.where(k_classes == i)[0]
		if len(duplicates) > 1:
			for j in range(len(duplicates)-1):
				count_cluster_cat[duplicates[0]] += count_cluster_cat[duplicates[j+1]]
				rows_deleted.append(duplicates[j+1])

	rows_deleted.sort()
	rows_deleted = rows_deleted[::-1]

	for i in rows_deleted:
		count_cluster_cat = np.delete(count_cluster_cat, i, axis=0)		
		k_classes = np.delete(k_classes, i, axis=0)

	# calculate metrics
	k = len(k_classes)
	tp = np.zeros(k)
	fp = np.zeros(k)
	fn = np.zeros(k)

	column_sum = np.sum(count_cluster_cat, axis = 0)
	row_sum = np.sum(count_cluster_cat, axis = 1)

	for i in range(k):
		tp[i] = count_cluster_cat[i][k_classes[i]]
		fp[i] = row_sum[i] - tp[i]
		fn[i] = column_sum[k_classes[i]] - tp[i]

	micro_precision = tp/(tp+fp)
	micro_recall = tp/(tp+fn)
	micro_fscore = (2 * micro_precision * micro_recall)/ (micro_precision + micro_recall)

	macro_precision = np.average(micro_precision)
	macro_recall = np.average(micro_recall)
	macro_fscore = np.average(micro_fscore)

	return macro_precision, macro_recall, macro_fscore, iterations


# *************************************************************************
# ******** read in data and transform it into numpy arrays ****************
# *************************************************************************

#  create variables
data = []
word_set = set()

# read in file
read_data('CA2data.txt')

#capture total no. of training instances
total_data_instances = len(data)							

#capture the no. of unique words in the training data
no_words = len(word_set)
# create a list of all words
word_list = list(word_set)

# create arrays which represent the training instances and the labels
data_array = np.zeros((total_data_instances,no_words), dtype = np.float)
data_label_char = np.chararray(total_data_instances, itemsize=20)
data_label_float = np.zeros(total_data_instances, dtype = np.float)

# transform data into numpy arrays
fill_arrays(total_data_instances, data, data_array, data_label_char, word_list, data_label_float)

# create array of unique classes
categories = np.unique(data_label_char)

# convert label array from string to floats
convert_label_array(data_label_char, categories, data_label_float, total_data_instances)

# normalizze the instance vectors
data_array = normalize_array(data_array)

# *************************************************************************
# ******************* run the k-means algorithm ***************************
# *************************************************************************

#  set the number of time to repeat the experiments
repeats = 20

# create arrays
results = np.zeros((4,repeats), dtype = np.float)
results_original = np.zeros((4,19), dtype = np.float)
results_closest = np.zeros((4,19), dtype = np.float)

# experiments for Q4 - selecting the mean as the cluster center
for y in range(2,21):
	for x in range(repeats):
		precision, recall, fscore, iterations = k_means(y, " ")
		results[0][x] = precision
		results[1][x] = recall
		results[2][x] = fscore
		results[3][x] = iterations
	# average the results across 20 runs of the algorithm
	results_original[:,y-2] = np.average(results, axis=1) 
print results_original


# experiments for Q5 - selecting the instance closest to the mean as the cluster centre
for y in range(2,21):
	for x in range(repeats):
		precision, recall, fscore, iterations = k_means(y, "closest_to_mean")
		results[0][x] = precision
		results[1][x] = recall
		results[2][x] = fscore
		results[3][x] = iterations
	# average the results across 20 runs of the algorithm
	results_closest[:,y-2] = np.average(results, axis=1)
print results_closest

# concatenate results and export to txt file
overall_results = np.concatenate([results_original, results_closest])
np.savetxt("results.txt", overall_results, delimiter=',')