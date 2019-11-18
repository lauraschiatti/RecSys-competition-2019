#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

"""
	data_manager.py: module for loading and preparing data. Also for displaying some statistics.
"""

import scipy.sparse as sps

DATASET_FOLDER = "dataset/"

# global vars
user_list = []
item_list = []
num_interactions = 0


def build_urm():
	global user_list
	global item_list
	global num_interactions

	print("Loading data_train ... ", end="\n")

	matrix_tuples = []

	with open(DATASET_FOLDER + 'data_train.csv', 'r') as file:  # read file's content
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

	URM = sps.coo_matrix((rating_list, (user_list, item_list)), shape=None) # (data, (row, col))
	URM = URM.tocsr()  # put in Compressed Sparse Row format

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


# Getters
# -------

# Get user_id list
def get_user_list_unique():
	list_unique = list(set(user_list)) # remove duplicates
	return list_unique

# Get item_id list
def get_item_list_unique():
	list_unique = list(set(item_list)) # remove duplicates
	return list_unique



# Build URM
URM = build_urm()

