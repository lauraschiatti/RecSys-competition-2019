#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Massimo Quadrana
"""

import numpy as np
from recommenders.BaseRecommender import BaseRecommender
from utils.DataIO import DataIO


# Recommends the top 10 most popular items to each user (highest number of interactions)
###### it simply recommends to a user the most popular items that the user has not previously consumed.

class TopPopRecommender(BaseRecommender):
    """Top Popular recommender"""

    RECOMMENDER_NAME = "TopPopRecommender"

    def __init__(self, URM_train):
        super(TopPopRecommender, self).__init__(URM_train)


    # model is the item popularity
    def fit(self):

        # Use np.ediff1d and NOT a sum done over the rows as there might be values other than 0/1
        self.item_pop = np.ediff1d(self.URM_train.tocsc().indptr)
        self.n_items = self.URM_train.shape[1]


    def _compute_item_score(self, user_id_array, items_to_compute = None):

        # Create a single (n_items, ) array with the item score, then copy it for every user

        if items_to_compute is not None:
            item_pop_to_copy = - np.ones(self.n_items, dtype=np.float32)*np.inf
            item_pop_to_copy[items_to_compute] = self.item_pop[items_to_compute].copy()
        else:
            item_pop_to_copy = self.item_pop.copy()

        item_scores = np.array(item_pop_to_copy, dtype=np.float32).reshape((1, -1))
        item_scores = np.repeat(item_scores, len(user_id_array), axis = 0)

        return item_scores


    def save_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {"item_pop": self.item_pop}

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)

        self._print("Saving complete")
