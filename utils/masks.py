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