#!/usr/local/bin/python3.6
# -*- coding: utf-8 -*-

"""
    evaluation.py: module for evaluating performance of recommenders.
"""

import numpy as np
import scipy.sparse as sps
import time, traceback

# Get test set relevant items for a given user
def get_relevant_items(user_id, URM_test):
    relevant_items = URM_test[user_id].indices

    return relevant_items

# Check whether recommended items are relevant
def get_is_relevant(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True) # compare elements in both arrays

    return is_relevant

# Precision: how many of the recommended items are relevant?
def precision(is_relevant):
    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)

    return precision_score

# Recall: how many of the relevant items I was able to recommend?
def recall(is_relevant, relevant_items):
    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]

    return recall_score

# Mean Average Precision
def MAP(is_relevant, relevant_items):
    # Cumulative sum: precision at k=1, at k=2, at k=3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

    return map_score


def evaluate_algorithm(URM_test, recommender_object, at=10):
    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_MAP = 0.0
    num_eval = 0

    # predict ratings for left-out ratings only (those in the URM_test)
    # and compute evaluation metrics on the results

    URM_test = sps.csr_matrix(URM_test)
    n_users = URM_test.shape[0]

    for user_id in range(n_users):

        if user_id % 1000 == 0:
            print("Evaluated user {} of {}".format(user_id, n_users))

        start_pos = URM_test.indptr[user_id]
        end_pos = URM_test.indptr[user_id + 1]

        if end_pos-start_pos>0:

            relevant_items = URM_test.indices[start_pos:end_pos]

            if len(relevant_items) > 0:
                recommended_items = recommender_object.recommend(user_id, at=at)
                num_eval+=1

                is_relevant = get_is_relevant(recommended_items, relevant_items)

                cumulative_precision += precision(is_relevant)
                cumulative_recall += recall(is_relevant, relevant_items)
                cumulative_MAP += MAP(is_relevant, relevant_items)

    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    cumulative_MAP /= num_eval

    print("Recommender performance is: Precision = {:.5f}, Recall = {:.5f}, MAP = {:.5f}".format(
        cumulative_precision, cumulative_recall, cumulative_MAP))

    result_dict = {
        "precision": cumulative_precision,
        "recall": cumulative_recall,
        "MAP": cumulative_MAP,
    }

    return result_dict

# Time to compute recommendations for a fixed group of users
def response_time(n_users_to_test, recommender):
    try:
        start_time = time.time()

        for user_id in range(n_users_to_test):
            recommender.recommend(user_id, at=5)

        end_time = time.time()

        print("Reasonable implementation speed is {:.2f} usr/sec".format(n_users_to_test / (end_time - start_time)))

    except Exception as e:
        print("Exception {}".format(str(e)))
        traceback.print_exc()