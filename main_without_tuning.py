#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

import numpy as np
from utils import data_manager
from utils import evaluation as eval
from utils import create_submission_file as create_csv
from utils import data_splitter
from utils import masks
from recommenders import RandomRecommender, TopPopRecommender, UserCFKNNRecommender, ItemCFKNNRecommender, \
    SLIM_BPR_Recommender,SLIMElasticNetRecommender, itemCBFKNNRecommender


# Build URM
# ---------

URM = data_manager.build_URM()

data_manager.get_statistics_URM(URM)

# Get 5% top popular items from the training data
five_perc_pop = data_manager.top_5_percept_popular_items(URM)
print("five_perc_pop", five_perc_pop, end='\n')

ICM = data_manager.build_ICM()

# Cold items, cold users and cold features
# ----------------------------------------

# URM, ICM = masks.refactor_URM_ICM(URM, ICM)


# ------------------------------------------ #
#   Train model without parameters tuning
# ------------------------------------------ #

# Train/test splitting
# --------------------

use_validation_set = False
k_out_value = 1  # Leave One Out (keep 1 interaction/user)
leave_random_out = True

# splitted_data = data_splitter.split_train_leave_k_out_user_wise(URM, k_out=k_out_value,
#                                                            use_validation_set=use_validation_set,
#                                                            leave_random_out=leave_random_out)

splitted_data = data_splitter.split_train_validation_random_holdout(URM, train_split=0.8)


if use_validation_set:
    URM_train, URM_validation, URM_test = splitted_data

else:
    URM_train, URM_test = splitted_data

SPLIT_URM_DICT = {
    "URM_train": URM_train,
    "URM_test": URM_test,
}

assert data_splitter.assert_disjoint_matrices(list(SPLIT_URM_DICT.values()))

data_manager.get_statistics_splitted_URM(SPLIT_URM_DICT)


recommender_list = [
    'RandomRecommender',
    'TopPopRecommender',
    # 'ItemCBFKNNRecommender',
    'UserCFKNNRecommender',
    'ItemCFKNNRecommender',
    'SLIM_BPR_Recommender',
    'SLIMElasticNetRecommender']

print('Recommender Systems: ')
for i, recomm_type in enumerate(recommender_list, start=1):
    print('{}. {}'.format(i, recomm_type))


# Fit the model
# -------------

while True:
    try:
        selected = int(input('\nSelect a recommender system: '.format(i)))
        recomm_type = recommender_list[selected-1]
        print('\n ... {} ... '.format(recomm_type))

        if recomm_type == 'RandomRecommender':
            recommender = RandomRecommender.RandomRecommender()
            recommender.fit(URM_train)

        elif recomm_type == 'TopPopRecommender':
            recommender = TopPopRecommender.TopPopRecommender()
            recommender.fit(URM_train)

        # Content-based filtering
        # elif recomm_type == 'ItemCBFKNNRecommender':
            # topK = 200
            # shrink = 10

            # recommender = itemCBFKNNRecommender.ItemCBFKNNRecommender(URM_train, ICM)
            # recommender.fit()


        # Collaborative filtering
        elif recomm_type == 'UserCFKNNRecommender':
            # recommender = UserCFKNNRecommender.UserCFKNNRecommender(URM_train)

            # MAP_per_k = []
            # for topK in [50, 100, 200]:
            #     print("topK = ", topK)
            #     for shrink in [10, 50, 100]:
            #         print("shrink = ", shrink)
            #
            #         recommender.fit(shrink=shrink, topK=topK)
            #         result_dict = eval.evaluate_algorithm(URM_test, recommender)
            #         MAP_per_k.append(result_dict["MAP"])

            topK = 200
            shrink = 10

            recommender = UserCFKNNRecommender.UserCFKNNRecommender(URM_train)
            recommender.fit(topK=topK, shrink=shrink)

        elif recomm_type == 'ItemCFKNNRecommender':
            # recommender = ItemCFKNNRecommender.ItemCFKNNRecommender(URM_train)

            # MAP_per_k = []
            # for topK in [50, 100, 200]:
            #     print("topK = ", topK)
            #     for shrink in [10, 50, 100]:
            #         print("shrink = ", shrink)
            #
            #         recommender.fit(shrink=shrink, topK=topK)
            #         result_dict = eval.evaluate_algorithm(URM_test, recommender)
            #         MAP_per_k.append(result_dict["MAP"])

            topK = 100
            shrink = 50
            recommender = ItemCFKNNRecommender.ItemCFKNNRecommender(URM_train)

            recommender.fit(topK=topK, shrink=shrink)

        # SLIM
        elif recomm_type == 'SLIM_BPR_Recommender':
            recommender = SLIM_BPR_Recommender.SLIM_BPR_Recommender(URM_train)
            recommender.fit()


        elif recomm_type == 'SLIMElasticNetRecommender':
            recommender = SLIMElasticNetRecommender.SLIMElasticNetRecommender(URM_train)
            recommender.fit()

        break

    except (ValueError, IndexError):
        print('Error. Please enter number between 1 and {}'.format(i))


# Evaluate model on left-out ratings (URM_test)
# ---------------------------------------------

eval.evaluate_algorithm(URM_test, recommender)


# Compute top-10 recommendations for each target user
# ---------------------------------------------------

predictions = input('\nCompute and save top10 predictions?: '
                    'y - Yes  n - No\n')

top_10_items = {}

if predictions == 'y':

    target_user_id_list = data_manager.get_target_users()

    for user_id in target_user_id_list[0:20]:

        item_list = ''
        for item in range(10):  # recommended_items
            item_list = recommender.recommend(user_id)

            top_10_items[user_id] = item_list  # .strip() # remove trailing space

    # Prints the nicely formatted dictionary
    # import pprint
    # pprint.pprint(top_10_items)

    # save predictions on csv file
    create_csv.create_csv(top_10_items, recomm_type)

