#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

import numpy as np
from utils import data_manager
from utils import create_submission_file as create_csv
from utils import data_splitter
from utils import masks
from utils.Evaluation.Evaluator import EvaluatorHoldout
from utils.KNN import  ItemKNNCFRecommender, UserKNNCFRecommender


# ------------------------------------------ #
#   Train models with parameters tuning
# ------------------------------------------ #

URM = data_manager.build_URM()

data_manager.get_statistics_URM(URM)

# Get 5% top popular items from the training data
five_perc_pop = data_manager.top_5_percept_popular_items(URM)
print("five_perc_pop", five_perc_pop, end='\n')

ICM = data_manager.build_ICM()

# Cold items, cold users and cold features
# ----------------------------------------

# URM, ICM = masks.refactor_URM_ICM(URM, ICM)


# Training
# --------

recommender_list = [
    'ItemKNNCFRecommender',
    'UserKNNCFRecommender',
    'itemKNNCF+P3alpha',
    'itemKNNCF+pureSVD',
    'itemKNNCF+TopPop',
    'SLIM_BPR']

print('Recommender Systems: ')
for i, recomm_type in enumerate(recommender_list, start=1):
    print('{}. {}'.format(i, recomm_type))

# Fit the model
# -------------

while True:
    try:
        selected = int(input('\nSelect a recommender system: '.format(i)))
        recomm_type = recommender_list[selected - 1]
        print('\n ... {} ... '.format(recomm_type))

        # Collaborative filtering
        if recomm_type == 'ItemKNNCFRecommender':
            #             from utils.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
            recommender_class = ItemKNNCFRecommender
            best_parameters, URM_train, URM_test = data_manager.parameter_tuning(URM, recommender_class=recommender_class)

            # Use an ItemKNNCF with the parameters we just learned
            recommender = ItemKNNCFRecommender.ItemKNNCFRecommender(URM_train)
            recommender.fit(**best_parameters)

            # Evaluate recommender
            evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
            result_dict, _ = evaluator_test.evaluateRecommender(recommender)

            print("result_dict MAP", result_dict[10]["MAP"])

            predictions = input('\nCompute and save top10 predictions?: '
                                'y - Yes  n - No\n')

            if predictions == 'y':
                # Train the model on the whole dataset using tuned params
                recommender = ItemKNNCFRecommender(URM)
                recommender.fit(**best_parameters)

                top_10_items = {}
                target_user_id_list = get_target_users()

                for user_id in target_user_id_list:
                    item_list = ''
                    for item in range(10):  # recommended_items
                        item_list = recommender.recommend(user_id, cutoff=10)
                        item_list = np.array(item_list)  # list to np.array

                        top_10_items[user_id] = item_list  # .strip() # remove trailing space

                # save predictions on csv file
                ccreate_csv(top_10_items, recomm_type)


        elif recomm_type == 'UserKNNCFRecommender':

            #             from utils.KNN.UserKNNCFRecommender import UserKNNCFRecommender
            recommender_class = UserKNNCFRecommender
            best_parameters, URM_train, URM_test = data_manager.parameter_tuning(URM, recommender_class=recommender_class)

            recommender = UserKNNCFRecommender(URM_train)
            recommender.fit(**best_parameters)

            # Evaluate recommender
            evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
            result_dict, _ = evaluator_test.evaluateRecommender(recommender)

            print("result_dict MAP", result_dict[10]["MAP"])

        # Linear combination of item-based models
        # take two matrices as inputs as well as the weights
        elif recomm_type == "itemKNNCF+P3alpha":

            best_parameters, URM_train, URM_test = parameter_tuning(URM, recommender_class=ItemKNNCFRecommender)

            itemKNNCF = ItemKNNCFRecommender(URM_train)
            itemKNNCF.fit(**best_parameters)

            #             from GraphBased.P3alphaRecommender import P3alphaRecommender

            P3alpha = P3alphaRecommender(URM_train)
            P3alpha.fit()

            hybridrecommender = ItemKNNSimilarityHybridRecommender(URM_train, itemKNNCF.W_sparse, P3alpha.W_sparse)
            hybridrecommender.fit(alpha=0.5)

            # Evaluate recommender
            evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
            result_dict, _ = evaluator_test.evaluateRecommender(hybridrecommender)

            print("result_dict MAP", result_dict[10]["MAP"])

        # Linear combination of predictions
        # In case of models with incompatible structure (e.g., ItemKNN with UserKNN or MF) you may ensemble the prediction values
        elif recomm_type == "itemKNNCF+pureSVD":

            best_parameters, URM_train, URM_test = parameter_tuning(URM, recommender_class=ItemKNNCFRecommender)

            itemKNNCF = ItemKNNCFRecommender(URM_train)
            itemKNNCF.fit(**best_parameters)

            pureSVD = PureSVDRecommender(URM_train)
            pureSVD.fit()

            hybridrecommender = ItemKNNScoresHybridRecommender(URM_train, itemKNNCF, pureSVD)
            hybridrecommender.fit(alpha=0.5)

            # Evaluate recommender
            evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
            result_dict, _ = evaluator_test.evaluateRecommender(hybridrecommender)

            print("result_dict MAP", result_dict[10]["MAP"])

        # User-wise hybrid
        # Models do not have the same accuracy for different user types.
        # Let's divide the users according to their profile length and then compare the recommendation quality we get from a CF model
        elif recomm_type == 'itemKNNCF+TopPop':  # poor results

            best_parameters, URM_train, URM_test = parameter_tuning(URM, recommender_class=ItemKNNCFRecommender)

            itemKNNCF = ItemKNNCFRecommender(URM_train)
            itemKNNCF.fit(**best_parameters)

            topPop = TopPop(URM_train)
            topPop.fit()

            recommender = ItemKNNScoresHybridRecommender(URM_train, itemKNNCF, topPop)
            recommender.fit(alpha=0.5)

            # Evaluate recommender
            evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
            result_dict, _ = evaluator_test.evaluateRecommender(recommender)

            print("result_dict MAP", result_dict[10]["MAP"])

        #         elif recomm_type == "SLIM_BPR":

        break

    except (ValueError, IndexError):
        print('Error. Please enter number between 1 and {}'.format(i))


