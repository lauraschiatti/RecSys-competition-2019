# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/04/18
@author: Maurizio Ferrari Dacrema
"""

from recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from utils.compute_similarity import check_matrix

class ItemKNNScoresHybridRecommender(BaseItemSimilarityMatrixRecommender):
    """ ItemKNNScoresHybridRecommender
    Hybrid of two prediction scores R = R1*alpha + R2*(1-alpha)

    """

    RECOMMENDER_NAME = "ItemKNNScoresHybridRecommender"

    def __init__(self, URM_train, Recommender_1, Recommender_2):
        super(ItemKNNScoresHybridRecommender, self).__init__(URM_train)

        self.URM_train = check_matrix(URM_train.copy(), 'csr')
        self.Recommender_1 = Recommender_1
        self.Recommender_2 = Recommender_2

    def fit(self, alpha=0.5):
        self.alpha = alpha

    def _compute_item_score(self, user_id_array, items_to_compute):
        item_weights_1 = self.Recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.Recommender_2._compute_item_score(user_id_array)

        item_weights = item_weights_1 * self.alpha + item_weights_2 * (1 - self.alpha)

        return item_weights