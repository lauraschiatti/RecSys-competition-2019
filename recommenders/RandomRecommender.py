#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

# Doesn't have anything to learn from the data
class RandomRecommender(object):

    def fit(self, URM_train):
        self.num_items = URM_train.shape[0]

    def recommend(self, user_id, at=10): # retrieve 10 items by
        recommended_items = np.random.choice(self.num_items, at)

        return recommended_items