#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

"""
	data_manager.py: module for loading and preparing data. Also for displaying some statistics.
"""

import scipy.sparse as sps
import numpy as np

dataset_dir = "dataset/"

# Interactions files (URM)
data_train = dataset_dir + "/data_train.csv"
data_target_users = dataset_dir + "/data_target_users_test.csv"

# Item content files (ICM)
data_ICM_asset = dataset_dir + "/data_ICM_asset.csv"  # description of the item (id)
data_ICM_price = dataset_dir + "/data_ICM_price.csv"  # price of each item (already normalized)
data_ICM_sub_class = dataset_dir + "/data_ICM_sub_class.csv"  # categorization of the item (number)


# global vars
user_list = []
item_list = []
num_interactions = 0

# -------------------------------------------
# User Rating Matrix from training data
# -------------------------------------------

def build_URM():
    global user_list, item_list, num_interactions

    print("\n ... Loading train data ... ", end="\n")

    matrix_tuples = []

    with open(data_train, 'r') as file:  # read file's content
        next(file)  # skip header row
        for line in file:
            if len(line.strip()) != 0: #  ignore lines with only whitespace
                num_interactions += 1

                # Create a tuple for each interaction (line in the file)
                matrix_tuples.append(row_split(line))

    # Separate user_id, item_id and rating
    user_list, item_list, rating_list = zip(*matrix_tuples)  # join tuples together (zip() to map values)

    # Create lists of all users, items and contents (ratings)
    user_list = list(user_list) # row
    item_list = list(item_list) # col
    rating_list = list(rating_list) # data

    URM = csr_sparse_matrix(rating_list, user_list, item_list)

    return URM


# Get statistics from interactions in the URM
# -------------------------------------------

def get_statistics_URM(URM):
    print("\n ... Statistics on URM ... ")

    print("No. of interactions in the URM is {}".format(num_interactions))

    user_list_unique = get_user_list_unique()
    item_list_unique = get_item_list_unique()

    n_unique_users = len(user_list_unique)
    n_unique_items = len(item_list_unique)

    n_users, n_items = URM.shape

    print("No. of unique items\t {}, No. of unique users\t {}".format(n_items, n_users))
    print("No. of items\t {}, No. of users\t {}".format(n_unique_items, n_unique_users))
    # print("\tMax ID items\t {}, Max ID users\t {}\n".format(max(item_list_unique), max(user_list_unique)))


use_validation_set = False
def get_statistics_splitted_URM(SPLIT_URM_DICT):

    print("\n ... Statistics on splitted URM ... ")

    n_users, n_items = SPLIT_URM_DICT["URM_train"].shape

    statistics_string = "Num items: {}\n" \
                        "Num users: {}\n" \
                        "Train \t\tinteractions {}, \tdensity {:.2E}\n".format(
        n_items,
        n_users,
        SPLIT_URM_DICT["URM_train"].nnz, compute_density(SPLIT_URM_DICT["URM_train"]))

    if use_validation_set:
        statistics_string += "Validation \tinteractions {}, \tdensity {:.2E}\n".format(
            SPLIT_URM_DICT["URM_validation"].nnz, compute_density(SPLIT_URM_DICT["URM_validation"]))

    statistics_string += "Test \t\tinteractions {}, \tdensity {:.2E}\n".format(
        SPLIT_URM_DICT["URM_test"].nnz, compute_density(SPLIT_URM_DICT["URM_test"]))

    print(statistics_string)


def split_train_validation_random_holdout(URM, train_split):
    number_interactions = URM.nnz  # number of nonzero values
    URM = URM.tocoo()  # Coordinate list matrix (COO)
    shape = URM.shape

    #  URM.row: user_list, URM.col: item_list, URM.data: rating_list

    # Sampling strategy: take random samples of data using a boolean mask
    train_mask = np.random.choice(
        [True, False],
        number_interactions,
        p=[train_split, 1 - train_split])  # train_perc for True, 1-train_perc for False

    URM_train = csr_sparse_matrix(URM.data[train_mask],
                                  URM.row[train_mask],
                                  URM.col[train_mask],
                                  shape=shape)

    test_mask = np.logical_not(train_mask)  # remaining samples
    URM_test = csr_sparse_matrix(URM.data[test_mask],
                                 URM.row[test_mask],
                                 URM.col[test_mask],
                                 shape=shape)

    return URM_train, URM_test

# -------------------------------------------------------------------------
# Build Item Content Matrix with three features: asset, price and sub-class
# -------------------------------------------------------------------------

def buildICM():
    # features = [‘asset’, ’price’, ’subclass’] info about products
    global user_list, item_list, num_interactions
#
#     print("\n ... Loading train data ... ", end="\n")
#
#     matrix_tuples = []
#
#     with open(data_train, 'r') as file:  # read file's content
#         next(file)  # skip header row
#         for line in file:
#             num_interactions += 1
#
#             # Create a tuple for each interaction (line in the file)
#             matrix_tuples.append(row_split(line))
#
#     # Separate user_id, item_id and rating
#     user_list, item_list, rating_list = zip(*matrix_tuples)  # join tuples together (zip() to map values)
#
#     # Convert values to list# Create lists of all users, items and contents (ratings)
#     user_list = list(user_list)
#     item_list = list(item_list)
#     content_list = list(content_list)
#     timestamp_list = list(timestamp_list)
#
#     return user_list, item_list, content_list, timestamp_list
#
#     n_items = URM.shape[1]
#     n_tags = max(tag_list_ICM) + 1
#     ICM_shape = (n_items, n_tags)
#
#     ones = np.ones(len(tag_list_ICM))
#
#     ICM = data.csr_sparse_matrix(ones, item_list_ICM, tag_list_ICM, ICM_shape)

# def get_statistics_ICM(self):
#
#     self._assert_is_initialized()
#
#     if len(self.dataReader_object.get_loaded_ICM_names()) > 0:
#
#         for ICM_name, ICM_object in self.SPLIT_ICM_DICT.items():
#             n_items, n_features = ICM_object.shape
#
#             statistics_string = "\tICM name: {}, Num features: {}, feature occurrences: {}, density {:.2E}".format(
#                 ICM_name,
#                 n_features,
#                 ICM_object.nnz,
#                 compute_density(ICM_object)
#             )
#
#             print(statistics_string)
#
#         print("\n")



def compute_density(URM):

    n_users, n_items = URM.shape
    n_interactions = URM.nnz

    # This avoids the fixed bit representation of numpy preventing
    # an overflow when computing the product
    n_items = float(n_items)
    n_users = float(n_users)

    if n_interactions == 0:
        return 0.0

    return n_interactions/(n_items*n_users)

# Getters
# -------

# Get all user_id list
def get_user_list_unique():
    list_unique = list(set(user_list))  # remove duplicates
    return list_unique


# Get item_id list
def get_item_list_unique():
    list_unique = list(set(item_list))  # remove duplicates
    return list_unique


# Get target user_id list
def get_target_users():
    target_user_id_list = []

    with open(data_target_users, 'r') as file:  # read file's content
        next(file)  # skip header row
        for line in file:
            # each line is a user_id
            target_user_id_list.append(int(line.strip()))  # remove trailing space

    return target_user_id_list


def row_split(row_string):
    # file format: 0,3568,1.0

    split = row_string.split(",")
    split[2] = split[2].replace("\n", "")

    split[0] = int(split[0])
    split[1] = int(split[1])
    split[2] = float(split[2])  # rating is a float

    result = tuple(split)
    return result


# Matrix Compressed Sparse Row format
# -----------------------------------

def csr_sparse_matrix(data, row, col, shape=None):
    csr_matrix = sps.coo_matrix((data, (row, col)), shape=shape)
    csr_matrix = csr_matrix.tocsr()


    return csr_matrix