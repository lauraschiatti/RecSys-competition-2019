#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

# Recommends to all users the items with highest average rating

class GlobalEffectsRecommender(object):

    def fit(self, URM_train):
        self.URM_train = URM_train

        # Average of all ratings, or global average
        global_average = np.mean(URM_train.data)
        # print("The global average is {:.2f}".format(global_average))

        # Subtract the average to all ratings
        URM_train_unbiased = URM_train.copy()
        URM_train_unbiased.data -= global_average

        # User Bias: average rating for each user
        # Remark: user bias is essential in case of rating prediction but not relevant in case of TopK recommendations.
        user_mean_rating = URM_train_unbiased.mean(axis=1)
        user_mean_rating = np.array(user_mean_rating).squeeze()
        # user_mean_rating = np.sort(user_mean_rating[user_mean_rating != 0.0])
        # data.plot_data(user_mean_rating, 'ro', 'User Mean Rating', 'User Bias', 'User Index')


        # In order to apply the user bias we have to change the rating value
        # in the URM_train_unbiased inner data structures
        # If we were to write:
        # URM_train_unbiased[user_id].data -= user_mean_rating[user_id]
        # we would change the value of a new matrix with no effect on the original data structure
        for user_id in range(len(user_mean_rating)):
            start_position = URM_train_unbiased.indptr[user_id]
            end_position = URM_train_unbiased.indptr[user_id + 1]

            URM_train_unbiased.data[start_position:end_position] -= user_mean_rating[user_id]


        # Item Bias: average rating for each item
        item_mean_rating = URM_train_unbiased.mean(axis=0)
        item_mean_rating = np.array(item_mean_rating).squeeze()
        # item_mean_rating = np.sort(item_mean_rating[item_mean_rating != 0])
        # data.plot_data(item_mean_rating, 'ro', 'Item Mean Rating', 'Item Bias', 'Item Index')

        self.best_rated_items = np.argsort(item_mean_rating)
        self.best_rated_items = np.flip(self.best_rated_items, axis=0)


    def recommend(self, user_id, at=10, remove_seen=True):
        # Sort the items by their itemBias and use the same recommendation principle as in TopPop
        if remove_seen:
            unseen_items_mask = np.in1d(self.best_rated_items, self.URM_train[user_id].indices,
                                        assume_unique=True, invert=True)

            unseen_items = self.best_rated_items[unseen_items_mask]
            recommended_items = unseen_items[0:at]

        else:
            recommended_items = self.best_rated_items[0:at]

        return recommended_items


