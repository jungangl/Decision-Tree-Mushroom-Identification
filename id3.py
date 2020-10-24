#!/usr/bin/python
# CIS 472/572 -- Programming Homework #1
# Starter code provided by Daniel Lowd, 1/25/2018
import math
import sys
import re
# Node class for the decision tree
import node
import numpy as np
train = None
varnames = None
test = None
testvarnames = None
root = None



# Computes entropy of Bernoulli distribution with arameter p
def entropy(p):
	# if p = 0 or 1 return entropy as 0
	ent = 0
	if 0 < p < 1:
		p1, p2 = p, 1 - p
		ent = -(p1 * math.log2(p1) + p2 * math.log2(p2))
	return ent
	


# Compute information gain for a particular split, given the counts
# py_pxi : number of occurences of y=1 with x_i=1 for all i=1 to n
# pxi : number of occurrences of x_i=1
# py : number of ocurrences of y=1
# total : total length of the data
def infogain(py_pxi, pxi, py, total):
	# Number of occurences of:
	# "xi = 1", "xi = 0", "y = 1", "y = 0"
	n_xi1 = pxi
	n_xi0 = total - n_xi1
	n_y1 = py
	n_y0 = total - n_y1
	# Number of occurences of:
	# "xi = 1, y = 1", "xi = 1, y = 0", "xi = 0, y = 1", "xi = 0, y = 0"
	n_xi1_y1 = py_pxi
	n_xi1_y0 = n_xi1 - n_xi1_y1
	n_xi0_y1 = n_y1 - n_xi1_y1
	n_xi0_y0 = n_y0 - n_xi1_y0
	# Compute the entropy before the split
	p = float(n_y1) / (n_y0 + n_y1)
	ent = entropy(p)
	# Compute the entropies after the split with xi = 0 and xi = 1
	p_xi0 = float(n_xi0_y1) / (n_xi0_y0 + n_xi0_y1)
	ent_xi0 = entropy(p_xi0)
	p_xi1 = float(n_xi1_y1) / (n_xi1_y0 + n_xi1_y1)
	ent_xi1 = entropy(p_xi1)
	# Compute the proportion for xi = 0 and xi = 1
	r_xi0 = n_xi0 / (n_xi0 + n_xi1)
	r_xi1 = n_xi1 / (n_xi0 + n_xi1)
	# Compute the gain
	gain = ent - (r_xi0 * ent_xi0 + r_xi1 * ent_xi1)  
	return gain



# For each variable, return the following count:
# py_pxi_list: number of occurences of y=1 with x_i=1
# pxi_list: number of occurrences of x_i=1
# py: number of ocurrences of y=1
# total: total length of the data
def count_number(data):
	data = np.array(data)
	y_data = data[:, -1]
	x_data = data[:, :-1]
	x_data_y1 = x_data[y_data == 1]

	py_pxi_list = np.sum(x_data_y1, axis = 0)
	pxi_list = np.sum(x_data, axis = 0)
	py = sum(y_data)
	total = len(y_data)
	return (py_pxi_list, pxi_list, py, total)



# Find the index of the best variable to split on, according to mutual information
def find_split(data):
	(py_pxi_list, pxi_list, py, total) = count_number(data)

	infogain_list = [infogain(py_pxi, pxi, py, total) for (py_pxi, pxi) in zip(py_pxi_list, pxi_list)]

	# Find the index of the largest infogain
	index = np.argmax(np.array(infogain_list))
	largest_gain = infogain_list[index]

	return index, largest_gain



# Partition data based on a given variable	
def partition_data(data, index):
	data = np.array(data)
	# Slice the data
	data_left = data[data[:, index] == 0, :]
	data_right = data[data[:, index] == 1, :]

	# Disable the coloumn of index by replacing it with a column of zeros
	# This column won't be used for as splits anymore.
	data_left[:, index] = 0
	data_right[:, index] = 0

	return (data_left.tolist(), data_right.tolist())



# Load data from a file
def read_data(filename):
	f = open(filename, 'r')
	p = re.compile(',')
	data = []
	header = f.readline().strip()
	varnames = p.split(header)
	namehash = {}
	for l in f:
		data.append([int(x) for x in p.split(l.strip())])
	return (data, varnames)



# Saves the model to a file.  Most of the work here is done in the
# node class.  This should work as-is with no changes needed.
def print_model(root, modelfile):
	f = open(modelfile, 'w+')
	root.write(f, 0)



# Build tree in a top-down manner, selecting splits until we hit a pure leaf or all splits look bad.
# Use a threshhold of 1e-5 to determine whether to return a split of leaf.
def build_tree(data, varnames):
	tree = None
	index, largest_gain = find_split(data)

	if largest_gain < 0.01:
		(py_pxi_list, pxi_list, py, total) = count_number(data)
		tree = node.Leaf(varnames, int(py/total > 0.5))
	else:
		(data_left, data_right) = partition_data(data, index)
		subtree_left = build_tree(data_left, varnames)
		subtree_right = build_tree(data_right, varnames)
		tree = node.Split(varnames, index, subtree_left, subtree_right)
	return tree



# "varnames" is a list of names, one for each variable
# "train" and "test" are lists of examples.
# Each example is a list of attribute values, where the last element in
# the list is the class value.
def loadAndTrain(trainS,testS,modelS):
	global train
	global varnames
	global test
	global testvarnames
	global root
	(train, varnames) = read_data(trainS)
	(test, testvarnames) = read_data(testS)
	modelfile = modelS

	root = build_tree(train, varnames)
	root.write(sys.stdout, 0)
	print_model(root, modelfile)
	# build_tree is the main function you'll have to implement, along with
    # any helper functions needed.  It should return the root node of the
    # decision tree.
	
	

def runTest():
	correct = 0
	# The position of the class label is the last element in the list.
	yi = len(test[0]) - 1
	for x in test:
		# Classification is done recursively by the node class.
        # This should work as-is.
		pred = root.classify(x)
		if pred == x[yi]:
			correct += 1
	acc = float(correct)/len(test)
	return acc	
	
	
	
# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
	if (len(argv) != 3):
		print('Usage: id3.py <train> <test> <model>')
		sys.exit(2)
	loadAndTrain(argv[0],argv[1],argv[2])

	acc = runTest()
	print("Accuracy: ",acc)

if __name__ == "__main__":
	main(sys.argv[1:])