#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time

from utils import masks, compute_similarity
from utils.data_manager import top_5_percept_popular_items


# SLIM with BPR
class SLIM_BPR_Recommender(object):
    """ SLIM_BPR recommender with cosine similarity and no shrinkage"""

    def __init__(self, URM):
        self.URM = URM
        # self.URM_mask = masks.get_warm_users_URM(self.URM)

        self.n_users, self.n_items = self.URM.shape

        # Initialize model: in the case of SLIM it works best to initialize S as zero
        self.similarity_matrix = np.zeros((self.n_items, self.n_items))

        # eligible users: users having at least one interaction
        self.eligible_users = []

        for user_id in range(self.n_users):

            start_pos = self.URM.indptr[user_id]
            end_pos = self.URM.indptr[user_id + 1]

            if len(self.URM.indices[start_pos:end_pos]) > 0:
                self.eligible_users.append(user_id)


    def sample_triplet(self):

        """
            Randomly samples a user and then samples randomly a seen and not seen item
            :return: user_id, pos_item_id, neg_item_id
        """

        user_id = self.sample_user()
        pos_item_id, neg_item_id = self.sample_item_pair(user_id)

        return user_id, pos_item_id, neg_item_id


    def sample_user(self):
        """
        Sample a user that has viewed at least one and not all items
        :return: user_id
        """

        while (True):
            # By randomly selecting a user in this way we could end up
            # with a user with no interactions
            # user_id = np.random.randint(0, self.n_users)

            user_id = np.random.choice(self.eligible_users)

            num_seen_items = self.URM[user_id].nnz

            if (num_seen_items > 0 and num_seen_items < self.n_items):
                return user_id


    def sample_item_pair(self, user_id):
        """
        Returns for the given user a random seen item and a random not seen item
        :param user_id:
        :return: pos_item_id, neg_item_id
        """

        # Get user seen items and choose one
        user_seen_items = self.URM[user_id].indices
        pos_item_id = user_seen_items[np.random.randint(0,len(user_seen_items))]

        # user_seen_items = self.URM[user_id, :].indices
        # pos_item_id = np.random.choice(user_seen_items)

        while(True):
            neg_item_id = np.random.randint(0, self.n_items)

            if(neg_item_id not in user_seen_items):
                return pos_item_id, neg_item_id

            # negItemSelected = False
            #
            # # It's faster to just try again then to build a mapping of the non-seen items
            # while (not negItemSelected):
            #     neg_item_id = np.random.randint(0, self.n_items)
            #
            #     if (neg_item_id not in user_seen_items):
            #         negItemSelected = True


    def update_factors(self, user_id, pos_item_id, neg_item_id):

        from scipy.special import expit

        user_seen_items = self.URM[user_id, :].indices

        # Calculate current predicted score
        x_i = self.similarity_matrix[pos_item_id, user_seen_items].sum()
        x_j = self.similarity_matrix[neg_item_id, user_seen_items].sum()

        # Gradient
        x_uij = x_i - x_j # prediction

        gradient = expit(-x_uij) # expit(x) = 1/(1+exp(-x))

        # Update similarities for all items

        # For positive item is PLUS logistic minus lambda*S
        self.similarity_matrix[pos_item_id, user_seen_items] += self.learning_rate * gradient
        self.similarity_matrix[pos_item_id, pos_item_id] = 0

        # For positive item is MINUS logistic minus lambda*S
        self.similarity_matrix[neg_item_id, user_seen_items] -= self.learning_rate * gradient
        self.similarity_matrix[neg_item_id, neg_item_id] = 0



    def epoch_iteration(self):

        # Get number of available interactions
        num_positive_interactions = int(self.URM.nnz * 0.01)

        start_time_epoch = time.time()
        start_time_batch = time.time()

        # Uniform user sampling without replacement
        for num_sample in range(num_positive_interactions):

            # sample triplets
            user_id, pos_item_id, neg_item_id = self.sample_triplet()

            # compute predictions, gradient and update model
            self.update_factors(user_id, pos_item_id, neg_item_id)


            if (time.time() - start_time_batch >= 30 or num_sample == num_positive_interactions - 1):
                print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Sample per second: {:.0f}".format(
                    num_sample,
                    100.0 * float(num_sample) / num_positive_interactions,
                    time.time() - start_time_batch,
                    float(num_sample) / (time.time() - start_time_epoch)))

                start_time_batch = time.time()


    def fit(self, learning_rate=1e-4, epochs=15):
        """
            Train SLIM wit BPR. If the model was already trained, overwrites matrix S
            :param epochs:
            :return: -
        """

        self.learning_rate = learning_rate

        start_time_train = time.time()

        for current_epoch in range(epochs):
            start_time_epoch = time.time()

            self.epoch_iteration()
            print("Epoch {} of {} complete in {:.2f} minutes".format(current_epoch + 1, epochs,
                                                                     float(time.time() - start_time_epoch) / 60))

        print("Train completed in {:.2f} minutes".format(float(time.time() - start_time_train) / 60))

        # The similarity matrix is learnt row-wise

        # To be used in the product URM*S must be transposed to be column-wise
        self.similarity_matrix = self.similarity_matrix.T

        self.similarity_matrix = compute_similarity.similarityMatrixTopK(self.similarity_matrix, k = 100)


    def recommend(self, user_id, at=None, exclude_seen=True, exclude_popular=True):

        # compute the scores using the dot product
        user_profile = self.URM[user_id]
        scores = user_profile.dot(self.similarity_matrix).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        if exclude_popular:
            recommended_items = self.filter_popular(ranking)
            return np.array(recommended_items) # list to np.array

        else:
            return ranking[:at]


    def filter_seen(self, user_id, scores):

        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]

        user_profile = self.URM.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores


    # Do not recommend 5% top popular items.
    def filter_popular(self, ranking, at=10):
        # get 5 % top popular items
        five_perc_pop = top_5_percept_popular_items(self.URM)

        i = 0
        recommended_items = []

        # Return 10 non-popular items to recommend
        while len(recommended_items) < at:

            # if the item in the ranking is not popular
            if ranking[i] not in five_perc_pop:
                # append to items to be recommended
                recommended_items.append(ranking[i])
            i += 1

        return recommended_items