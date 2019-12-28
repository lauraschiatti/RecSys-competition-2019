#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

import traceback
import numpy as np
from utils.data_manager import build_URM, build_ICM, get_statistics_URM, get_target_users
from utils.evaluation import evaluate_algorithm
from utils.Evaluation.Evaluator import EvaluatorHoldout
from utils.ParameterTuning.hyperparameter_search import runParameterSearch_Collaborative, runParameterSearch_Content
from utils.DataIO import DataIO
from utils.create_submission_file import create_csv
from utils.data_splitter import split_train_validation_random_holdout, split_train_leave_k_out_user_wise

######################################################################
##########                                                  ##########
##########                  PURE COLLABORATIVE              ##########
##########                                                  ##########
######################################################################
# Non-Personalized
from recommenders.RandomRecommender import RandomRecommender
from recommenders.TopPopRecommender import TopPopRecommender
# Global effects not implemented for implicit ratings


# KNN
from recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from recommenders.KNN.ItemKNNSimilarityHybridRecommender import ItemKNNSimilarityHybridRecommender

# Graph-based
from recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
# from GraphBased.RP3betaRecommender import RP3betaRecommender

# KNN machine learning
# from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
# from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender

# Matrix Factorization
from recommenders.PureSVDRecommender import PureSVDRecommender
# from MatrixFactorization.IALSRecommender import IALSRecommender
# from MatrixFactorization.NMFRecommender import NMFRecommender
# from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython,\
#     MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython


######################################################################
##########                                                  ##########
##########                  PURE CONTENT BASED              ##########
##########                                                  ##########
######################################################################
from recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender

######################################################################
##########                                                  ##########
##########                 HYBRID RECOMMENDERS              ##########
##########                                                  ##########
######################################################################
from recommenders.Hybrid.CFW_D_Similarity_Linalg import CFW_D_Similarity_Linalg
from recommenders.Hybrid.ItemKNNScoresHybridRecommender import ItemKNNScoresHybridRecommender

# Build URM, ICM and UCM
# ----------------------

URM_all = build_URM()
ICM_all = build_ICM()
# get_statistics_URM(URM_all)

# Cold items, cold users and cold features

# NOTE:
# Usually to deal with cold items you use a content-collaborative hybrid
# URM, ICM = masks.refactor_URM_ICM(URM, ICM)

# The issue with cold users and cold items is that a personalized collaborative recommender
# is not able to model them.
# Even if you train a model like a matrix factorization or even an itemKNN and
# you get a recommendation list, those are just random recommendations.

# In order to overcome this, you have to look for other models that allow you to provide
# a meaningful result for cold items and users (e.g., TopPop, content-based …)
# and build a hybrid. The easiest solution is to average those models or to switch among
# them depending on the user of interest. You may find some hints in the practice sessions for
# hybrid and collaborative boosted FW

# Top-10 recommenders
cutoff = 10  # k recommended_items

# URM train/validation/test splitting
# -----------------------------------
k_out = 1

# URM_train, URM_test = split_train_validation_random_holdout(URM_all, train_split=0.8)
# URM_train, URM_validation = split_train_validation_random_holdout(URM_train, train_split=0.9)

URM_train, URM_test = split_train_leave_k_out_user_wise(URM_all,
                                                        k_out=k_out,
                                                        use_validation_set=False,
                                                        leave_random_out=True)

URM_train, URM_validation = split_train_leave_k_out_user_wise(URM_train,
                                                              k_out=k_out,
                                                              use_validation_set=False,
                                                              leave_random_out=True)

# Recommenders
# ------------

# Non-personalized recommenders
non_personalized_list = [
    RandomRecommender,
    TopPopRecommender
]

# Graph-based recommenders
graph_algorithm_list = [
    # P3alphaRecommender,
    # RP3betaRecommender,
]

# Collaborative recommenders
collaborative_algorithm_list = [
    ItemKNNCFRecommender,
    UserKNNCFRecommender,
    #     MatrixFactorization_BPR_Cython,
    # MatrixFactorization_FunkSVD_Cython,
    PureSVDRecommender,
    # SLIM_BPR_Cython,
    # SLIMElasticNetRecommender
]

# Content-based recommenders
content_algorithm_list = [
    ItemKNNCBFRecommender
]

# Hybrid recommenders
hybrid_algorithm_list = [
    ItemKNNSimilarityHybridRecommender,  # Linear combination of item-based models
    CFW_D_Similarity_Linalg,  # regression problem using linalg solver
    ItemKNNScoresHybridRecommender  # Linear combination of predictions
]

recommender_list = [
    # Non-personalized
    RandomRecommender,
    TopPopRecommender,

    # Graph-based recommenders
    # P3alphaRecommender,
    # RP3betaRecommender,

    # Collaborative recommenders
    ItemKNNCFRecommender,
    UserKNNCFRecommender,

    #     MatrixFactorization_BPR_Cython,
    # MatrixFactorization_FunkSVD_Cython,
    PureSVDRecommender,
    # SLIM_BPR_Cython,
    # SLIMElasticNetRecommender,

    # Content-based recommenders
    ItemKNNCBFRecommender,

    # Hybrid recommenders
    ItemKNNSimilarityHybridRecommender,
    CFW_D_Similarity_Linalg,
    ItemKNNScoresHybridRecommender
]

# Best hyperparameters found by tuning
# ------------------------------------

best_parameters_list = {
    'RandomRecommender': {},
    'TopPopRecommender': {},

    'MatrixFactorization_BPR_Cython': {'sgd_mode': 'adagrad', 'epochs': 1500, 'num_factors': 177, 'batch_size': 4,
                                       'positive_reg': 2.3859950782265896e-05,
                                       'negative_reg': 7.572911338047984e-05,
                                       'learning_rate': 0.0005586331284886803},

    'ItemKNNCFRecommender': {'topK': 5, 'shrink': 1000, 'similarity': 'cosine', 'normalize': True,
                             'feature_weighting': 'none'},

    'ItemKNNCBFRecommender': {'topK': 983, 'shrink': 18, 'similarity': 'cosine', 'normalize': True,
                              'feature_weighting': 'none'}

}


################################################################################################################

# User-wise hybrid
# ----------------

# Models do not have the same accuracy for different user types.
# Let's divide the users according to their profile length and then compare
# the recommendation quality we get from a CF model


# # TopPop
# topPop = TopPopRecommender(URM_train)
# topPop.fit()
#
# # Hybrid: ItemKNNCF + pureSVD
#
# # ItemKNNCFRecommender
# itemKNNCF = ItemKNNCFRecommender(URM_train)
# best_parameters_ItemKNNCF = {'topK': 9, 'shrink': 47, 'similarity': 'cosine', 'normalize': True,
#                              'feature_weighting': 'none'}
# itemKNNCF.fit(**best_parameters_ItemKNNCF)
#
# # PureSVD
# pureSVD = PureSVDRecommender(URM_train)
# best_parameters_PureSVD = {'num_factors': 350}
# pureSVD.fit(**best_parameters_PureSVD)
#
# itemKNN_scores_hybrid = ItemKNNScoresHybridRecommender(URM_train, itemKNNCF, pureSVD)
# best_parameters = {'alpha': 0.9}
# itemKNN_scores_hybrid.fit(**best_parameters)
#
# # profile for all users (URM_all)
#
# profile_length = np.ediff1d(URM_all.indptr)
# block_size = int(len(profile_length) * 0.21)
# n_users, n_items = URM_train.shape
# num_groups = int(np.ceil(n_users / block_size))
# sorted_users = np.argsort(profile_length)
#
# MAP_topPop_per_group = []
# MAP_itemKNN_scores_hybrid_per_group = []
#
# for group_id in range(0, num_groups):
#     start_pos = group_id * block_size
#     end_pos = min((group_id + 1) * block_size, len(profile_length))
#
#     users_in_group = sorted_users[start_pos:end_pos]
#
#     users_in_group_p_len = profile_length[users_in_group]
#
#     print("Group {}, average p.len {:.2f}, min {}, max {}".format(group_id,
#                                                                   users_in_group_p_len.mean(),
#                                                                   users_in_group_p_len.min(),
#                                                                   users_in_group_p_len.max()))
#
#     users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
#     users_not_in_group = sorted_users[users_not_in_group_flag]
#
#     evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[cutoff], ignore_users=users_not_in_group)
#
#     results, _ = evaluator_test.evaluateRecommender(topPop)
#     MAP_topPop_per_group.append(results[cutoff]["MAP"])
#
#     results, _ = evaluator_test.evaluateRecommender(itemKNN_scores_hybrid)
#     MAP_itemKNN_scores_hybrid_per_group.append(results[cutoff]["MAP"])
#
# print("plotting.....")
#
# import matplotlib.pyplot as pyplot
#
# pyplot.plot(MAP_topPop_per_group, label="topPop")
# pyplot.plot(MAP_itemKNN_scores_hybrid_per_group, label="ItemKNNCF + pureSVD")
# pyplot.ylabel('MAP')
# pyplot.xlabel('User Group')
# pyplot.legend()
# pyplot.show()


################################################################################################################

# --- generate predictions --- #

# Train models on the whole dataset
# TopPop
topPop = TopPopRecommender(URM_all)
topPop.fit()

# Hybrid: ItemKNNCF + pureSVD
# ItemKNNCFRecommender
itemKNNCF = ItemKNNCFRecommender(URM_all)
best_parameters_ItemKNNCF = {'topK': 9, 'shrink': 47, 'similarity': 'cosine', 'normalize': True,
                             'feature_weighting': 'none'}
itemKNNCF.fit(**best_parameters_ItemKNNCF)

# PureSVD
pureSVD = PureSVDRecommender(URM_all)
best_parameters_PureSVD = {'num_factors': 350}
pureSVD.fit(**best_parameters_PureSVD)

itemKNN_scores_hybrid = ItemKNNScoresHybridRecommender(URM_all, itemKNNCF, pureSVD)
best_parameters_itemKNN_scores_hybrid = {'alpha': 0.9}
itemKNN_scores_hybrid.fit(**best_parameters_itemKNN_scores_hybrid)


# profile for all users (URM_all)

profile_length = np.ediff1d(URM_all.indptr)
block_size = int(len(profile_length) * 0.21)
n_users, n_items = URM_all.shape
num_groups = int(np.ceil(n_users / block_size))
sorted_users = np.argsort(profile_length)

users_by_group = []

for group_id in range(0, num_groups):
    start_pos = group_id * block_size
    end_pos = min((group_id + 1) * block_size, len(profile_length))

    users_in_group = sorted_users[start_pos:end_pos]

    users_in_group_p_len = profile_length[users_in_group]

    print("Group {} with users_in_group {}, average p.len {:.2f}, min {}, max {}".format(group_id,
                                                                                         len(users_in_group),
                                                                                         users_in_group_p_len.mean(),
                                                                                         users_in_group_p_len.min(),
                                                                                         users_in_group_p_len.max()))

    # Users by group
    users_by_group.append(users_in_group)


# Generate predictions
top_10_items = {}
user_id_array = get_target_users()
items = []

for user_id in user_id_array:

    # TopPop for users with fewer interactions
    if user_id in users_by_group[0] or user_id in users_by_group[1]:
        # print("user_id: {}, group: 0 or 1, topPop".format(user_id))
        item_list = topPop.recommend(user_id,
                                     cutoff=cutoff,
                                     remove_seen_flag=True,
                                     remove_top_pop_flag=True)
    else:
        # print("user_id: {}, group: 2, 3, 4, itemKNN_scores_hybrid".format(user_id))
        item_list = itemKNN_scores_hybrid.recommend(user_id,
                                                    cutoff=cutoff,
                                                    remove_seen_flag=True,
                                                    remove_top_pop_flag=True)

    items.append(np.array(item_list))
    # item_list = np.array(item_list)  # list to np.array
    # top_10_items[user_id] = item_list


# print("top_10_items ... ")
# import pprint
# pprint.pprint(items)
# save predictions on csv file
create_csv(user_id_array, items, None)

exit(0)


################################################################################################################

print('\nRecommender Systems: ')
for i, recomm_type in enumerate(recommender_list, start=1):
    print('{}. {}'.format(i, recomm_type.RECOMMENDER_NAME))

while True:
    try:
        selected = int(input('\nSelect a recommender system: '.format(i)))
        recommender_class = recommender_list[selected - 1]
        print('\n ... {} ... '.format(recommender_class.RECOMMENDER_NAME))

        # Hyperparams tuning
        # ----------------------

        apply_hyperparams_tuning = False

        if apply_hyperparams_tuning:
            # best_parameters = read_data_split_and_search(recommender_class)

            metric_to_optimize = "MAP"

            evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[cutoff])
            evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[cutoff, cutoff + 5])
            evaluator_validation_earlystopping = EvaluatorHoldout(URM_train, cutoff_list=[cutoff], exclude_seen=False)
            output_folder_path = "result_experiments/"

            n_cases = 8  # 2
            n_random_starts = 5  # int(n_cases / 3)

            save_model = "no"
            allow_weighting = True  # provides better results
            similarity_type_list = ["cosine"]

            ICM_name = "ICM_all"

            output_file_name_root = "{}_metadata.zip".format(recommender_class.RECOMMENDER_NAME)

            if recommender_class in [non_personalized_list, collaborative_algorithm_list]:
                try:
                    runParameterSearch_Collaborative(recommender_class=recommender_class,
                                                     URM_train=URM_train,
                                                     metric_to_optimize=metric_to_optimize,
                                                     evaluator_validation=evaluator_validation,
                                                     evaluator_test=evaluator_test,
                                                     evaluator_validation_earlystopping=evaluator_validation_earlystopping,
                                                     output_folder_path=output_folder_path,
                                                     n_cases=n_cases,
                                                     n_random_starts=n_random_starts,
                                                     save_model=save_model,
                                                     allow_weighting=allow_weighting,
                                                     similarity_type_list=similarity_type_list)

                    if recommender_class in [ItemKNNCFRecommender, UserKNNCFRecommender]:
                        similarity_type = similarity_type_list[0]  # KNN Recommenders on similarity_type
                        output_file_name_root = "{}_{}_metadata.zip".format(recommender_class.RECOMMENDER_NAME,
                                                                            similarity_type)

                except Exception as e:
                    print("On recommender {} Exception {}".format(recommender_class, str(e)))
                    traceback.print_exc()

            elif recommender_class in [content_algorithm_list, hybrid_algorithm_list]:
                try:
                    runParameterSearch_Content(recommender_class=recommender_class,
                                               URM_train=URM_train,
                                               ICM_object=ICM_all,
                                               ICM_name=ICM_name,
                                               n_cases=n_cases,
                                               n_random_starts=n_random_starts,
                                               save_model=save_model,
                                               evaluator_validation=evaluator_validation,
                                               evaluator_test=evaluator_test,
                                               metric_to_optimize=metric_to_optimize,
                                               output_folder_path=output_folder_path,
                                               allow_weighting=allow_weighting,
                                               similarity_type_list=similarity_type_list)

                    similarity_type = similarity_type_list[0]  # KNN Recommenders on similarity_type
                    output_file_name_root = "{}_{}_{}_metadata.zip".format(recommender_class.RECOMMENDER_NAME,
                                                                           ICM_name, similarity_type)

                except Exception as e:
                    print("On recommender {} Exception {}".format(recommender_class, str(e)))
                    traceback.print_exc()

                # Load best_parameters for training
                data_loader = DataIO(folder_path=output_folder_path)
                search_metadata = data_loader.load_data(output_file_name_root)
                best_parameters = search_metadata["hyperparameters_best"]  # dictionary with all the fit parameters
                print("best_parameters {}".format(best_parameters))

        else:
            try:
                best_parameters = best_parameters_list[recommender_class.RECOMMENDER_NAME]

            except Exception as e:
                print("best_parameters not found on recommender {} Exception {}".format(recommender_class, str(e)))
                traceback.print_exc()

        # Fit the recommender with the hyperparameters we just learned
        # -------------------------------------------------------

        if recommender_class in non_personalized_list:
            recommender = recommender_class(URM_train)
            recommender.fit()

        elif recommender_class in content_algorithm_list:
            recommender = recommender_class(URM_train, ICM_all)  # todo: ICM_all or ICM_train?
            recommender.fit(**best_parameters)

        elif recommender_class is ItemKNNSimilarityHybridRecommender:
            # Hybrid: ItemKNNCF + P3alpha
            itemKNNCF = ItemKNNCFRecommender(URM_train)
            best_parameters = {'topK': 9, 'shrink': 47, 'similarity': 'cosine', 'normalize': True,
                               'feature_weighting': 'none'}
            itemKNNCF.fit(**best_parameters)

            P3alpha = P3alphaRecommender(URM_train)
            P3alpha.fit()

            recommender = recommender_class(URM_train, itemKNNCF.W_sparse, P3alpha.W_sparse)
            best_parameters = {'alpha': 0.7}
            recommender.fit(**best_parameters)

            # Hybrid: ItemKNNCF + itemKNNCBF
            # itemKNNCBF = ItemKNNCBFRecommender(URM_train, ICM_all)
            # best_parameters = {'topK': 983, 'shrink': 18, 'similarity': 'cosine', 'normalize': True,
            #                    'feature_weighting': 'none'}
            # itemKNNCBF.fit(**best_parameters)
            #
            # recommender = recommender_class(URM_all, itemKNNCF.W_sparse, itemKNNCBF.W_sparse)
            # best_parameters = {'alpha': 0.8}
            # recommender.fit(**best_parameters)

        elif recommender_class is CFW_D_Similarity_Linalg:
            # feature weighting techniques
            itemKNNCF = ItemKNNCFRecommender(URM_train)
            best_parameters = {'topK': 9, 'shrink': 47, 'similarity': 'cosine', 'normalize': True,
                               'feature_weighting': 'none'}
            itemKNNCF.fit(**best_parameters)

            W_sparse_CF = itemKNNCF.W_sparse

            # hyperparams tuning
            # if recommender_class is CFW_D_Similarity_Linalg:
            #     hyperparameters_range_dictionary = {}
            #     hyperparameters_range_dictionary["topK"] = Integer(5, 1000)
            #     hyperparameters_range_dictionary["add_zeros_quota"] = Real(low=0, high=1, prior='uniform')
            #     hyperparameters_range_dictionary["normalize_similarity"] = Categorical([True, False])
            #
            #     recommender_input_args = SearchInputRecommenderArgs(
            #         CONSTRUCTOR_POSITIONAL_ARGS=[URM_train, ICM_all, W_sparse_CF],
            #         CONSTRUCTOR_KEYWORD_ARGS={},
            #         FIT_POSITIONAL_ARGS=[],
            #         FIT_KEYWORD_ARGS={}
            #     )

            # output_folder_path = "result_experiments/"
            #
            # import os
            #
            # # If directory does not exist, create
            # if not os.path.exists(output_folder_path):
            #     os.makedirs(output_folder_path)
            #
            # n_cases = 10
            # metric_to_optimize = "MAP"
            #
            # # Clone data structure to perform the fitting with the best hyperparameters on train + validation data
            # recommender_input_args_last_test = recommender_input_args.copy()
            # recommender_input_args_last_test.CONSTRUCTOR_POSITIONAL_ARGS[0] = URM_train + URM_validation
            #
            # parameterSearch.search(recommender_input_args,
            #                        recommender_input_args_last_test=recommender_input_args_last_test,
            #                        parameter_search_space=hyperparameters_range_dictionary,
            #                        n_cases=n_cases,
            #                        n_random_starts=int(n_cases / 3),
            #                        save_model="no",
            #                        output_folder_path=output_folder_path,
            #                        output_file_name_root=recommender_class.RECOMMENDER_NAME,
            #                        metric_to_optimize=metric_to_optimize
            #                        )
            #

            # Weighted Content-based similarity
            recommender = recommender_class(URM_train, ICM_all, W_sparse_CF)
            recommender.fit()

        elif recommender_class is ItemKNNScoresHybridRecommender:
            # Hybrid: ItemKNNCF + pureSVD
            itemKNNCF = ItemKNNCFRecommender(URM_train)
            best_parameters = {'topK': 9, 'shrink': 47, 'similarity': 'cosine', 'normalize': True,
                               'feature_weighting': 'none'}
            itemKNNCF.fit(**best_parameters)

            pureSVD = PureSVDRecommender(URM_train)
            best_parameters = {'num_factors': 350}
            pureSVD.fit(**best_parameters)

            recommender = recommender_class(URM_train, itemKNNCF, pureSVD)
            best_parameters = {'alpha': 0.9}
            recommender.fit(**best_parameters)

        else:
            recommender = recommender_class(URM_train)
            recommender.fit(**best_parameters)

        # Evaluate model
        evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[cutoff])
        result_dict, _ = evaluator_test.evaluateRecommender(recommender)

        print("{} result_dict MAP {}".format(recommender_class.RECOMMENDER_NAME, result_dict[cutoff]["MAP"]))

        # Generate predictions
        # --------------------

        predictions = input('\nCompute and save top10 predictions?: y - Yes  n - No\n')

        if predictions == 'y':

            # Train the model on the whole dataset using tuned params
            # -------------------------------------------------------

            if recommender_class in non_personalized_list:
                recommender = recommender_class(URM_all)
                recommender.fit()

            elif recommender_class in content_algorithm_list:
                recommender = recommender_class(URM_all, ICM_all)
                recommender.fit(**best_parameters)

            elif recommender_class is ItemKNNSimilarityHybridRecommender:
                itemKNNCF = ItemKNNCFRecommender(URM_all)
                best_parameters = {'topK': 9, 'shrink': 47, 'similarity': 'cosine', 'normalize': True,
                                   'feature_weighting': 'none'}
                itemKNNCF.fit(**best_parameters)

                P3alpha = P3alphaRecommender(URM_all)
                P3alpha.fit()

                recommender = recommender_class(URM_all, itemKNNCF.W_sparse, P3alpha.W_sparse)
                best_parameters = {'alpha': 0.7}
                recommender.fit(**best_parameters)

                # itemKNNCBF = ItemKNNCBFRecommender(URM_all, ICM_all)
                # best_parameters = {'topK': 983, 'shrink': 18, 'similarity': 'cosine', 'normalize': True,
                #                    'feature_weighting': 'none'}
                # itemKNNCBF.fit(**best_parameters)
                #
                # recommender = recommender_class(URM_all, itemKNNCF.W_sparse, itemKNNCBF.W_sparse)
                # best_parameters = {'alpha': 0.8}
                # recommender.fit(**best_parameters)

            elif recommender_class is CFW_D_Similarity_Linalg:
                itemKNNCF = ItemKNNCFRecommender(URM_all)
                best_parameters = {'topK': 14, 'shrink': 20, 'similarity': 'cosine', 'normalize': True,
                                   'feature_weighting': 'BM25'}
                itemKNNCF.fit(**best_parameters)

                recommender = recommender_class(URM_all, ICM_all, itemKNNCF.W_sparse)
                recommender.fit()

            elif recommender_class is ItemKNNScoresHybridRecommender:

                itemKNNCF = ItemKNNCFRecommender(URM_all)
                best_parameters = {'topK': 14, 'shrink': 20, 'similarity': 'cosine', 'normalize': True,
                                   'feature_weighting': 'BM25'}
                itemKNNCF.fit(**best_parameters)

                pureSVD = PureSVDRecommender(URM_all)
                best_parameters = {'num_factors': 350}
                pureSVD.fit(**best_parameters)

                recommender = recommender_class(URM_all, itemKNNCF, pureSVD)
                best_parameters = {'alpha': 0.9}
                recommender.fit(**best_parameters)

            else:
                recommender = recommender_class(URM_all)
                recommender.fit(**best_parameters)

            user_id_array = get_target_users()
            item_list = recommender.recommend(user_id_array,
                                              cutoff=cutoff,
                                              remove_seen_flag=True,
                                              remove_top_pop_flag=True)

            create_csv(user_id_array, item_list, recommender_class.RECOMMENDER_NAME)

        break

    except (ValueError, IndexError):
        print('Error. Please enter number between 1 and {}'.format(i))
