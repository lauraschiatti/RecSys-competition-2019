#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time

from utils import masks, compute_similarity


# SLIM with BPR
class SLIM_BPR_Recommender(object):
    """ SLIM_BPR recommender with cosine similarity and no shrinkage"""

    def __init__(self, URM_train):
        self.URM_train = URM_train

        self.URM_mask = masks.get_warm_users_URM(self.URM_train)

        self.n_users, self.n_items = self.URM_mask.shape

        # Initialize model: in the case of SLIM it works best to initialize S as zero
        self.similarity_matrix = np.zeros((self.n_items, self.n_items))

        # Initialize similarity with random values and zero-out diagonal
        # self.S = np.random.random((self.n_items, self.n_items)).astype('float32')
        # self.S[np.arange(self.n_items), np.arange(self.n_items)] = 0

        # Eligible users: users having at least one interaction
        self.eligible_users = []

        for user_id in range(self.n_users):

            start_pos = self.URM_mask.indptr[user_id]
            end_pos = self.URM_mask.indptr[user_id + 1]

            if len(self.URM_mask.indices[start_pos:end_pos]) > 0:
                self.eligible_users.append(user_id)

    # Randomly sample the triplets (user, positive_item, negative_item)
    def sample_triplet(self):

        # By randomly selecting a user in this way we could end up
        # with a user with no interactions
        # user_id = np.random.randint(0, n_users)

        user_id = np.random.choice(self.eligible_users)

        # Get user seen items and choose one
        user_seen_items = self.URM_mask[user_id, :].indices
        pos_item_id = np.random.choice(user_seen_items)

        negItemSelected = False

        # It's faster to just try again then to build a mapping of the non-seen items
        while (not negItemSelected):
            neg_item_id = np.random.randint(0, self.n_items)

            if (neg_item_id not in user_seen_items):
                negItemSelected = True

        return user_id, pos_item_id, neg_item_id

    def epoch_iteration(self):

        # Get number of available interactions
        numPositiveIteractions = int(self.URM_mask.nnz * 0.01)

        start_time_epoch = time.time()
        start_time_batch = time.time()

        # Uniform user sampling without replacement
        for num_sample in range(numPositiveIteractions):

            # Sample triplets (user, positive_item, negative_item)
            # ---------------

            user_id, positive_item_id, negative_item_id = self.sample_triplet()

            user_seen_items = self.URM_mask[user_id, :].indices

            # Prediction
            # ----------

            x_i = self.similarity_matrix[positive_item_id, user_seen_items].sum()
            x_j = self.similarity_matrix[negative_item_id, user_seen_items].sum()

            # Gradient: depends on the objective function: RMSE, BPR
            # ---------

            x_ij = x_i - x_j

            # The original BPR paper uses the logarithm of the sigmoid of x_ij, whose derivative is the following
            gradient = 1 / (1 + np.exp(x_ij))

            # Update model
            # -------------

            learning_rate = 1e-3

            # In SLIM there's just one parameter to update, the similarity matrix
            self.similarity_matrix[positive_item_id, user_seen_items] += learning_rate * gradient
            self.similarity_matrix[positive_item_id, positive_item_id] = 0

            self.similarity_matrix[negative_item_id, user_seen_items] -= learning_rate * gradient
            self.similarity_matrix[negative_item_id, negative_item_id] = 0

            if (time.time() - start_time_batch >= 30 or num_sample == numPositiveIteractions - 1):
                print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Sample per second: {:.0f}".format(
                    num_sample,
                    100.0 * float(num_sample) / numPositiveIteractions,
                    time.time() - start_time_batch,
                    float(num_sample) / (time.time() - start_time_epoch)))

                start_time_batch = time.time()

    def fit(self, learning_rate=0.01, epochs=10):

        self.learning_rate = learning_rate
        self.epochs = epochs

        for current_epoch in range(self.epochs):
            start_time_epoch = time.time()

            self.epoch_iteration()
            print("Epoch {} of {} complete in {:.2f} minutes".format(current_epoch + 1, epochs,
                                                                     float(time.time() - start_time_epoch) / 60))

        self.similarity_matrix = self.similarity_matrix.T

        self.similarity_matrix = compute_similarity.similarityMatrixTopK(self.similarity_matrix, k=100)

    def recommend(self, user_id, at=None, exclude_seen=True):
        # compute the scores using the dot product
        user_profile = self.URM_train[user_id]
        scores = user_profile.dot(self.similarity_matrix).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen(self, user_id, scores):

        start_pos = self.URM_train.indptr[user_id]
        end_pos = self.URM_train.indptr[user_id + 1]

        user_profile = self.URM_train.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores

# todo: check ../master/SLIM_BPR/SLIM_BPR.py