#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

import numpy as np

# Recommends to all users the most popular items, those with the highest number of interactions (ratings)
class TopPopRecommender(object):
    # model is the item popularity
    def fit(self, URM_train):
        self.URM_train = URM_train

        item_popularity = (URM_train > 0).sum(axis=0)
        item_popularity = np.array(item_popularity).squeeze()

        # We are not interested in sorting the popularity value,
        # but to order the items according to it
        self.popular_items = np.argsort(item_popularity)
        self.popular_items = np.flip(self.popular_items, axis=0)

    def recommend(self, user_id, at=10, remove_seen=True):
        if remove_seen: #  always remove seen items if your purpose is to recommend "new" ones
            unseen_items_mask = np.in1d(self.popular_items, self.URM_train[user_id].indices,
                                      assume_unique=True, invert=True)

            unseen_items = self.popular_items[unseen_items_mask]
            recommended_items = unseen_items[0:at]

        else:
            recommended_items = self.popular_items[0:at]

        return recommended_items

