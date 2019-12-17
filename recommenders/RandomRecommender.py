# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# import numpy as np
#
# # Recommend at random items to each user
#
# class RandomRecommender(object):
#
#     RECOMMENDER_NAME = "RandomRecommender"
#
#     def fit(self, URM_train):
#         self.num_items = URM_train.shape[0]
#
#     def recommend(self, user_id, at=10): # retrieve 10 items by
#         recommended_items = np.random.choice(self.num_items, at)
#
#         return recommended_items
#
#




# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Massimo Quadrana
"""

import numpy as np
from recommenders.BaseRecommender import BaseRecommender
from utils.DataIO import DataIO


# Recommend at random items to each user
class RandomRecommender(BaseRecommender):
    """Random recommender"""

    RECOMMENDER_NAME = "RandomRecommender"

    def __init__(self, URM_train):
        super(RandomRecommender, self).__init__(URM_train)


    def fit(self, random_seed=42):
        np.random.seed(random_seed)
        self.n_items = self.URM_train.shape[1]


    def _compute_item_score(self, user_id_array, items_to_compute = None):

        # Create a random block (len(user_id_array), n_items) array with the item score

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.n_items), dtype=np.float32)*np.inf
            item_scores[:, items_to_compute] = np.random.rand(len(user_id_array), len(items_to_compute))

        else:
            item_scores = np.random.rand(len(user_id_array), self.n_items)

        return item_scores



    def save_model(self, folder_path, file_name = None):

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {}

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save = data_dict_to_save)

        self._print("Saving complete")