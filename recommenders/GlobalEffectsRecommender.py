#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

# Recommends to all users the items with highest average rating

class GlobalEffectsRecommender(object):
    # todo: does not work for implicit ratings
    # with implicit ratings
    def fit(self, URM_train):
        self.URM_train = URM_train

        # 1) global average: average of all ratings
        mu = np.mean(self.URM_train.data[self.URM_train.data != 0])
        # remove mu from the URM (subs mu to all ratings)
        URM_train_unbiased = URM_train.copy()
        URM_train_unbiased.data -= mu

        # Initialize the biases
        n_users, n_items = URM_train.shape

        # 2) user average bias: average rating for each user
        user_bias = np.zeros(n_users)

        # 3) item average bias: average rating for each item
        item_bias = np.zeros(n_items)


        # 4) precompute the item ranking
        self.best_rated_items = np.argsort(item_bias)
        self.best_rated_items = np.flip(self.best_rated_items, axis=0)


    def recommend(self, user_id, at=10, remove_seen=False):
        # Sort the items by their item_bias and use the same recommendation principle as in TopPop
        user_seen_items = self.URM_train[user_id].indices

        if remove_seen:
            unseen_items_mask = np.in1d(self.best_rated_items, user_seen_items,
                                        assume_unique=True, invert=True)

            unseen_items = self.best_rated_items[unseen_items_mask]
            recommended_items = unseen_items[0:at]

        else:
            recommended_items = self.best_rated_items[0:at]

        return recommended_items


