#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import utils.compute_similarity as cs
from utils.data_manager import top_5_percept_popular_items

class ItemCFKNNRecommender(object):
    
    def __init__(self, URM):
        self.URM = URM
        
            
    def fit(self, topK=50, shrink=100, normalize=True, similarity="cosine"):
        
        similarity_object = cs.Compute_Similarity_Python(self.URM, shrink=shrink, 
                                                  topK=topK, normalize=normalize, 
                                                  similarity = similarity)
        
        self.W_sparse = similarity_object.compute_similarity()

        
    def recommend(self, user_id, at=None, exclude_seen=True, exclude_popular=True):
        # compute the scores using the dot product
        user_profile = self.URM[user_id]
        scores = user_profile.dot(self.W_sparse).toarray().ravel()

        if exclude_seen:
            scores = self.filter_seen(user_id, scores)

        # rank items
        ranking = scores.argsort()[::-1] # numpy.ndarray

        if exclude_popular:
            recommended_items = self.filter_popular(ranking)
            return np.array(recommended_items) # list to np.array

        else:
            return ranking[:at]
    
    
    def filter_seen(self, user_id, scores):

        start_pos = self.URM.indptr[user_id]
        end_pos = self.URM.indptr[user_id+1]

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

