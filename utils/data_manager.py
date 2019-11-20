#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

"""
	data_manager.py: module for loading and preparing data. Also for displaying some statistics.
"""

import scipy.sparse as sps
import numpy as np

dataset_dir = "dataset/"
data_train = "dataset/data_train.csv"
data_target_users = "dataset/data_target_users_test.csv"


# global vars
# -----------

user_list = []
item_list = []
num_interactions = 0


# Build User Rating Matrix from training data
# -------------------------------------------

def build_URM():
	global user_list, item_list, num_interactions

	print("\n ... Loading train data ... ", end="\n")

	matrix_tuples = []

	with open(data_train, 'r') as file:  # read file's content
		next(file) # skip header row
		for line in file:
			num_interactions += 1

			# Create a tuple for each interaction (line in the file)
			matrix_tuples.append(row_split(line, is_URM=True))


	# Separate user_id, item_id and rating
	user_list, item_list, rating_list = zip(*matrix_tuples)  # join tuples together (zip() to map values)

	# Convert values to list
	user_list = list(user_list)
	item_list = list(item_list)
	rating_list = list(rating_list)

	URM = csr_sparse_matrix(rating_list, user_list, item_list)

	return URM


def row_split(row_string, is_URM):
	# file format: 0,3568,1.0

	split = row_string.split(",")
	split[2] = split[2].replace("\n", "")

	split[0] = int(split[0])
	split[1] = int(split[1])

	if is_URM == True:
		split[2] = float(split[2])  # rating is a float
	elif is_URM == False:
		split[2] = str(split[2])  # tag is a string, not a float like the rating


	result = tuple(split)
	return result


# Matrix Compressed Sparse Row format
# -----------------------------------

def csr_sparse_matrix(data, row, col, shape=None):
	csr_matrix = sps.coo_matrix((data, (row, col)), shape=shape)
	csr_matrix = csr_matrix.tocsr()

	return csr_matrix


# Get statistics from interactions in the URM
# -------------------------------------------

def interactions_statistics():
	print("\n ... Statistics on URM ... ")

	print("No. of interactions in the URM is {}".format(num_interactions))

	user_list_unique = get_user_list_unique()
	item_list_unique = get_item_list_unique()

	num_users = len(user_list_unique)
	num_items = len(item_list_unique)

	print("No. of items\t {}, No. of users\t {}".format(num_items, num_users))
	print("Max ID items\t {}, Max ID users\t {}\n".format(max(item_list_unique), max(user_list_unique)))
	print("Average interactions per user {:.2f}".format(num_interactions / num_users))
	print("Average interactions per item {:.2f}\n".format(num_interactions / num_items))

	# sparsity = zero-valued elements / total number of elements
	print("Sparsity {:.2f} %\n".format((1 - float(num_interactions) / (num_items * num_users)) * 100))


# Train/test data splitting
# -------------------------

def train_test_holdout(URM, train_perc):
	number_interactions = URM.nnz  # number of nonzero values
	URM = URM.tocoo() # Coordinate list matrix (COO)
	shape = URM.shape

	#  URM.row: user_list, URM.col: item_list, URM.data: rating_list

	# Sampling strategy: take random samples of data using a boolean mask
	train_mask = np.random.choice(
					[True, False],
				  	number_interactions, p=[train_perc, 1-train_perc]) # train_perc for True, 1-train_perc for False

	URM_train = csr_sparse_matrix(URM.data[train_mask], URM.row[train_mask], URM.col[train_mask], shape=shape)

	test_mask = np.logical_not(train_mask) # remaining samples
	URM_test = csr_sparse_matrix(URM.data[test_mask], URM.row[test_mask], URM.col[test_mask], shape=shape)

	return URM_train, URM_test



# Getters
# -------

# Get all user_id list
def get_user_list_unique():
	list_unique = list(set(user_list)) # remove duplicates
	return list_unique

# Get item_id list
def get_item_list_unique():
	list_unique = list(set(item_list)) # remove duplicates
	return list_unique

# Get target user_id list
def get_target_users():
	target_user_id_list = []


	with open(data_target_users, 'r') as file:  # read file's content
		next(file)  # skip header row
		for line in file:
			# each line is a user_id
			target_user_id_list.append(int(line.strip())) # remove trailing space

	return target_user_id_list

