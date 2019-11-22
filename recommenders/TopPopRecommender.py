#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

import numpy as np

# Recommends the top 10 most popular items to each user (highest number of interactions)

###### it simply recommends to a user the most popular items that the user has not previously consumed.

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

        # print("Number of items with zero interactions {}".
        #       format(np.sum(item_popularity == 0)))

    def recommend(self, user_id, at=10, remove_seen=True):
        if remove_seen: #  always remove seen items if your purpose is to recommend "new" ones

            # seen items: those the user already interacted with
            user_seen_items = self.URM_train[user_id].indices
            unseen_items_mask = np.in1d(self.popular_items, user_seen_items,
                                        assume_unique=True, invert=True)

            unseen_items = self.popular_items[unseen_items_mask]
            recommended_items = unseen_items[0:at]

        else:
            recommended_items = self.popular_items[0:at]

        return recommended_items

