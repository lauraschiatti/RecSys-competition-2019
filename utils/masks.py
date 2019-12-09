#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

import scipy.sparse as sps
import numpy as np

# warm items and users arrays
warm_users_mask = []
warm_users = []
warm_items_mask = []
warm_items = []


def get_warm_items_URM(URM):
    global warm_items_mask, warm_items
    warm_items_mask = np.ediff1d(URM.tocsc().indptr) > 0
    warm_items = np.arange(URM.shape[1])[warm_items_mask]

    URM_all = URM[:, warm_items]

    return URM_all

def get_warm_users_URM(URM):

    # NOTE: The train (or test) data may contain a lot of low rating interactions (<= threshold),
    # In reality we want to recommend items rated in a positive way (> threshold ),
    # so let's build a new data set with positive interactions only

    # Remove 0 rating interactions (0.0 values) from the URM_train (or URM_test),
    # now .data contains only positive interactions

    global warm_users_mask, warm_users
    warm_users_mask = np.ediff1d(URM.indptr) > 0
    warm_users = np.arange(URM.shape[0])[warm_users_mask]

    URM_all = URM[warm_users, :]
    return URM_all

def get_cold_users_URM(URM):
    global warm_users_mask
    cold_users_mask = np.logical_not(warm_users_mask)
    cold_users = np.arange(URM.shape[0])[cold_users_mask]

    URM_all = URM[cold_users, :]
    return URM_all



def refactor_URM_ICM(URM, ICM):

    print("URM.shape", URM.shape)
    print("ICM.shape", ICM.shape)

    # Keep the warm_items and warm_users array, we might need them in future
    # warm_items_mask = np.ediff1d(URM.tocsc().indptr) > 0
    # warm_items = np.arange(URM.shape[1])[warm_items_mask]
    # URM = URM[:, warm_items]
    # warm_users_mask = np.ediff1d(URM.tocsr().indptr) > 0
    # warm_users = np.arange(URM.shape[0])[warm_users_mask]
    # URM = URM[warm_users, :]

    # Keep only warm_items and warm_features in the ICM
    # ICM = ICM[warm_items, :]
    # ICM = ICM.tocsr()
    #
    # warm_features_mask = np.ediff1d(ICM.tocsc().indptr) > 0
    # warm_features = np.arange(ICM.shape[1])[warm_features_mask]
    #
    # ICM = ICM[:, warm_features]
    # ICM = ICM.tocsr()

    # There could be items with no features
    nofeatures_items_mask = np.ediff1d(ICM.tocsr().indptr) < 3 #<=0
    nofeatures_items_mask.sum()

    # We might not remove them in some cases, but we will do it for our comparison
    warm_items_mask_2 = np.ediff1d(ICM.tocsr().indptr) > 0
    warm_items_2 = np.arange(ICM.shape[0])[warm_items_mask_2]

    ICM = ICM[warm_items_2, :]
    ICM = ICM.tocsr()

    # Now we have to remove cold items and users from the URM
    URM = URM[:, warm_items_2]
    URM = URM.tocsr()

    warm_users_mask_2 = np.ediff1d(URM.tocsr().indptr) > 0
    warm_users_2 = np.arange(URM.shape[0])[warm_users_mask_2]

    URM = URM[warm_users_2, :]
    URM = URM.tocsr()

    print("ICM.shape without warm_items", ICM.shape)
    print("URM.shape without warm_items", URM.shape)

    return URM, ICM