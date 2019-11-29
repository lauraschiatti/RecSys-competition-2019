#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import utils.compute_similarity as cs

class UserCFKNNRecommender(object):

    def __init__(self, URM):
        self.URM = URM

    def fit(self, topK=50, shrink=100, normalize=True, similarity="cosine"):
        similarity_object = cs.Compute_Similarity_Python(self.URM.T, shrink=shrink,
                                                      topK=topK, normalize=normalize,
                                                      similarity=similarity)

        self.W_sparse = similarity_object.compute_similarity()


    def recommend(self, user_id, at=10, exclude_seen=True):
        # compute the scores using the dot product

        scores = self.W_sparse[user_id, :].dot(self.URM).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1]

        return ranking[:at]

    def filter_seen(self, user_id, scores):
        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id + 1]

        user_profile = self.URM.indices[start_pos:end_pos]

        scores[user_profile] = -np.inf

        return scores