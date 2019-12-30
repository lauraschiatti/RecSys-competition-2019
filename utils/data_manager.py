#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

"""
	data_manager.py: module for loading and preparing data. Also for displaying some statistics.
"""

import scipy.sparse as sps
import numpy as np
from utils.compute_similarity import check_matrix

dataset_dir = "dataset/"

# Interactions files (URM)
data_train = dataset_dir + "/data_train.csv"
data_target_users = dataset_dir + "/data_target_users_test.csv"

# Item content files (ICM)
data_ICM_asset = dataset_dir + "/data_ICM_asset.csv"  # description of the item (id)
data_ICM_price = dataset_dir + "/data_ICM_price.csv"  # price of each item (already normalized)
data_ICM_sub_class = dataset_dir + "/data_ICM_sub_class.csv"  # categorization of the item (number)

# Item content files (UCM)
data_UCM_age = dataset_dir + "/data_UCM_age.csv"  # age of each user (already normalized)
data_UCM_region = dataset_dir + "/data_UCM_region.csv"  # region of each user (already normalized)

# global vars
user_list = []
item_list = []
n_interactions = 0
n_users = 0
n_items = 0
n_subclass = 0
n_regions = 0


# -------------------------------------------
# User Rating Matrix from training data
# -------------------------------------------

def build_URM():
    global user_list, item_list, n_interactions

    matrix_tuples = []

    with open(data_train, 'r') as file:  # read file's content
        next(file)  # skip header row
        for line in file:
            if len(line.strip()) != 0:  # ignore lines with only whitespace
                n_interactions += 1

                # Create a tuple for each interaction (line in the file)
                matrix_tuples.append(row_split(line))

    # Separate user_id, item_id and rating
    user_list, item_list, rating_list = zip(*matrix_tuples)  # join tuples together (zip() to map values)

    # Create lists of all users, items and contents (ratings)
    user_list = list(user_list)  # row
    item_list = list(item_list)  # col
    rating_list = list(rating_list)  # data

    URM = csr_sparse_matrix(rating_list, user_list, item_list)

    print("URM built!")
    print(URM[1:3, :].todense())
    print("\n")

    return URM


# Get statistics from interactions in the URM
# -------------------------------------------

def get_statistics_URM(URM):
    print("\n ... Statistics on URM ... ")
    global n_users, n_items

    print("No. of interactions in the URM is {}".format(n_interactions))

    user_list_unique = get_user_list_unique()
    item_list_unique = get_item_list_unique()

    n_unique_users = len(user_list_unique)
    n_unique_items = len(item_list_unique)

    n_users, n_items = URM.shape

    print("No. of unique items\t {}, No. of unique users\t {}".format(n_items, n_users))
    print("No. of items\t {}, No. of users\t {}".format(n_unique_items, n_unique_users))
    # print("\tMax ID items\t {}, Max ID users\t {}\n".format(max(item_list_unique), max(user_list_unique)))


def get_statistics_splitted_URM(SPLIT_URM_DICT):
    use_validation_set = False

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


def compute_density(URM):
    n_users, n_items = URM.shape
    n_interactions = URM.nnz

    # This avoids the fixed bit representation of numpy preventing
    # an overflow when computing the product
    n_items = float(n_items)
    n_users = float(n_users)

    if n_interactions == 0:
        return 0.0

    return n_interactions / (n_items * n_users)


# -------------------------------------------------------------------------
# Build Item Content Matrix with three features: asset, price and sub-class
# -------------------------------------------------------------------------

def build_ICM():
    # features = [‘asset’, ’price’, ’subclass’] info about products
    global n_subclass

    # Load subclass data
    matrix_tuples = []

    with open(data_ICM_sub_class, 'r') as file:  # read file's content
        next(file)  # skip header row
        for line in file:
            n_subclass += 1

            # Create a tuple for each interaction (line in the file)
            matrix_tuples.append(row_split(line))

    # Separate user_id, item_id and rating
    item_list, class_list, col_list = zip(*matrix_tuples)  # join tuples together (zip() to map values)

    # Convert values to list# Create lists of all users, items and contents (ratings)
    item_list_icm = list(item_list)
    class_list_icm = list(class_list)
    col_list_icm = np.zeros(len(col_list))

    # Number of items that are in the subclass list
    num_items = max(item_list_icm) + 1
    ICM_shape = (num_items, 1)
    ICM_subclass = csr_sparse_matrix(class_list_icm, item_list_icm, col_list_icm, shape=ICM_shape)

    # Load price data
    matrix_tuples = []
    n_prices = 0

    with open(data_ICM_price, 'r') as file:  # read file's content
        next(file)  # skip header row
        for line in file:
            n_prices += 1

            # Create a tuple for each interaction (line in the file)
            matrix_tuples.append(row_split(line))

    # Separate user_id, item_id and rating
    item_list, col_list, price_list = zip(*matrix_tuples)  # join tuples together (zip() to map values)

    # Convert values to list# Create lists of all users, items and contents (ratings)
    item_list_icm = list(item_list)
    col_list_icm = list(col_list)
    price_list_icm = list(price_list)

    ICM_price = csr_sparse_matrix(price_list_icm, item_list_icm, col_list_icm)

    # Load asset data
    matrix_tuples = []
    n_assets = 0

    with open(data_ICM_asset, 'r') as file:  # read file's content
        next(file)  # skip header row
        for line in file:
            n_assets += 1

            # Create a tuple for each interaction (line in the file)
            matrix_tuples.append(row_split(line))

    # Separate user_id, item_id and rating
    item_list, col_list, asset_list = zip(*matrix_tuples)  # join tuples together (zip() to map values)

    # Convert values to list# Create lists of all users, items and contents (ratings)
    item_list_icm = list(item_list)
    col_list_icm = list(col_list)
    asset_list_icm = list(asset_list)

    ICM_asset = csr_sparse_matrix(asset_list_icm, item_list_icm, col_list_icm)

    ICM_all = sps.hstack([ICM_price, ICM_asset, ICM_subclass], format='csr')

    # item_feature_ratios(ICM_all)

    print("ICM built!")
    print(ICM_all[1:3, :].todense())
    print("\n")

    return ICM_all


def build_UCM(URM):
    # features = [‘asset’, ’price’, ’subclass’] info about products

    # Load subclass data
    matrix_tuples = []

    with open(data_UCM_age, 'r') as file:  # read file's content
        next(file)  # skip header row
        for line in file:
            # Create a tuple for each interaction (line in the file)
            matrix_tuples.append(row_split(line))

    # Separate user_id and age
    user_list, age_list, col_list = zip(*matrix_tuples)  # join tuples together (zip() to map values)

    # Convert values to list# Create lists of all users and ages
    user_list_icm = list(user_list)
    age_list_icm = list(age_list)
    col_list_icm = np.zeros(len(col_list))

    # Number of items that are in the subclass list
    num_users = URM.shape[0]
    UCM_shape = (num_users, 1)
    UCM_age = csr_sparse_matrix(user_list_icm, age_list_icm, col_list_icm, shape=UCM_shape)

    # Load region data
    matrix_tuples = []
    global n_regions

    with open(data_UCM_region, 'r') as file:  # read file's content
        next(file)  # skip header row
        for line in file:
            # Create a tuple for each interaction (line in the file)
            matrix_tuples.append(row_split(line))

    # Separate user_id, item_id and rating
    user_list, region_list, col_list = zip(*matrix_tuples)  # join tuples together (zip() to map values)

    # Convert values to list# Create lists of all users, items and contents (ratings)
    user_list_icm = list(user_list)
    region_list_icm = list(region_list)

    n_regions = max(region_list_icm) + 1

    UCM_shape = (num_users, n_regions)

    ones = np.ones(len(region_list_icm))
    UCM_region = sps.coo_matrix((ones, (user_list_icm, region_list_icm)), shape = UCM_shape)
    UCM_region = UCM_region.tocsr()

    UCM_all = sps.hstack([UCM_age, UCM_region], format='csr')

    # item_feature_ratios(ICM_all)

    print("UCM built!")
    print(UCM_all[1:3, :].todense())
    print("\n")


    return UCM_all


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


# Get user_id seen items
def get_user_seen_items(user_id, URM):
    # seen items: those the user already interacted with
    user_seen_items = URM[user_id].indices

    return user_seen_items


# Get interactions of a given user_id (row in URM)
def get_user_profile(URM, user_id):
    start_user_position = URM.indptr[user_id]
    end_user_position = URM.indptr[user_id + 1]

    user_profile = URM.indices[start_user_position:end_user_position]

    # or interactions = URM[user_id, :]
    return user_profile


# Get users that have no Train items
def perc_user_no_item_train(URM_train):
    user_no_item_train = np.sum(np.ediff1d(URM_train.indptr) == 0)

    if user_no_item_train != 0:
        print("Warning: {} ({:.2f} %) of {} users have no Train items \n".format(user_no_item_train,
                                                                                 user_no_item_train / n_users * 100,
                                                                                 n_users))


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


# Get top 10% popular items from the training data
# over all items
# ---------------------------------------------------

def top_5_percept_popular_items(URM):
    # This is appropriate in cases where users can discover these items on their own,
    # and may not find these recommendations useful

    # print("\n ... Item popularity ... ")4
    item_popularity = (URM > 0).sum(axis=0)
    item_popularity = np.array(item_popularity).squeeze()
    item_popularity = np.sort(item_popularity)  # sorted array

    n_items = URM.shape[1]

    ten_percent = int(n_items / 5)
    ten_percent_popular_items = item_popularity[-ten_percent].mean()
    # print("Average per-item interactions for the top 10% popular items {:.2f}".
    #       format(ten_percent_popular_items))

    # Number of cold items
    # print("Number of items with zero interactions (cold items) {}".
    #       format(np.sum(item_popularity == 0)))

    item_popularity = (URM > 0).sum(axis=0)
    item_popularity = np.array(item_popularity).squeeze()

    # We are not interested in sorting the popularity value,
    # but to order the items according to it
    popular_items = np.argsort(item_popularity)  # sorted array indices
    popular_items = np.flip(popular_items, axis=0)  # reverse order of elements along the given axis

    ten_perc_pop = popular_items[0:int(ten_percent_popular_items)]

    return ten_perc_pop


def item_feature_ratios(ICM):
    # Features per item
    ICM = sps.csr_matrix(ICM)
    features_per_item = np.ediff1d(ICM.indptr)  # differences between consecutive elements of an array.
    print("\nFeatures Per Item: {}".format(features_per_item))

    # Items per feature
    ICM = sps.csc_matrix(ICM)
    items_per_feature = np.ediff1d(ICM.indptr)
    print("Items Per Feature: {}\n".format(items_per_feature))
