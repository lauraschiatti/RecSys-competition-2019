#!/usr/bin/env python3
#  -*- coding: utf-8 -*-


import os, traceback
from functools import partial

from utils.data_manager import build_URM
from utils.Evaluation.Evaluator import EvaluatorHoldout
from utils.ParameterTuning.parameter_search import runParameterSearch_Collaborative
from utils.DataIO import DataIO
from utils import create_submission_file as create_csv
from utils.data_splitter import split_train_validation_random_holdout
from utils import masks


######################################################################
##########                                                  ##########
##########        DATA LOADING AND PREPROCESSING            ##########
##########                                                  ##########
######################################################################

# from Data_manager.Movielens1M.Movielens1MReader import Movielens1MReader
# from Data_manager.DataSplitter_k_fold_stratified import DataSplitter_Warm_k_fold

# dataset_object = Movielens1MReader()
# dataSplitter = DataSplitter_Warm_k_fold(dataset_object)
# dataSplitter.load_data()
# URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()

# Build URM
# ---------

URM = build_URM()

# URM statistics
# --------------

# data_manager.get_statistics_URM(URM)

# URM train/validation/test splitting
# -------------------------------

URM_train, URM_test = split_train_validation_random_holdout(URM, train_split=0.8)
URM_train, URM_validation = split_train_validation_random_holdout(URM_train, train_split=0.9)

output_folder_path = "result_experiments/"

# If directory does not exist, create
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

######################################################################
##########                                                  ##########
##########                  PURE COLLABORATIVE              ##########
##########                                                  ##########
######################################################################
# from Base.NonPersonalizedRecommender import TopPop, Random, GlobalEffects

# KNN
from utils.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from utils.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
# from GraphBased.P3alphaRecommender import P3alphaRecommender
# from GraphBased.RP3betaRecommender import RP3betaRecommender

# KNN machine learning
# from SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
# from SLIM_ElasticNet.SLIMElasticNetRecommender import SLIMElasticNetRecommender

# Matrix Factorization
# from MatrixFactorization.PureSVDRecommender import PureSVDRecommender
# from MatrixFactorization.IALSRecommender import IALSRecommender
# from MatrixFactorization.NMFRecommender import NMFRecommender
# from MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_BPR_Cython,\
#     MatrixFactorization_FunkSVD_Cython, MatrixFactorization_AsySVD_Cython


######################################################################
##########                                                  ##########
##########                  PURE CONTENT BASED              ##########
##########                                                  ##########
######################################################################
# from KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender



######################################################################
##########                                                  ##########
##########      TRAINING, EVALUATION AND PREDICTIONS        ##########
##########                                                  ##########
######################################################################

evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10, 15])


n_cases = 8
metric_to_optimize = "MAP"
runParameterSearch_Collaborative_partial = partial(runParameterSearch_Collaborative,
                                                   URM_train=URM_train,
                                                   metric_to_optimize=metric_to_optimize,
                                                   n_cases=n_cases,
                                                   save_model="no",
                                                   evaluator_validation_earlystopping=evaluator_validation,
                                                   evaluator_validation=evaluator_validation,
                                                   evaluator_test=evaluator_test,
                                                   output_folder_path=output_folder_path)

collaborative_algorithm_list = [
    # Random,
    # TopPop,
    # P3alphaRecommender,
    # RP3betaRecommender,
    ItemKNNCFRecommender,
    UserKNNCFRecommender,
    # MatrixFactorization_BPR_Cython,
    # MatrixFactorization_FunkSVD_Cython,
    # PureSVDRecommender,
    # SLIM_BPR_Cython,
    # SLIMElasticNetRecommender
]

print('\nRecommender Systems: ')
for i, recomm_type in enumerate(collaborative_algorithm_list, start=1):
    print('{}. {}'.format(i, recomm_type.RECOMMENDER_NAME))


# from Utils.PoolWithSubprocess import PoolWithSubprocess
# import multiprocessing
#
# pool = PoolWithSubprocess(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
# resultList = pool.map(runParameterSearch_Collaborative_partial, collaborative_algorithm_list)
# pool.close()
# pool.join()


while True:
    try:
        selected = int(input('\nSelect a recommender system: '.format(i)))
        recomm_type = collaborative_algorithm_list[selected - 1]
        print('\n ... {} ... '.format(recomm_type.RECOMMENDER_NAME))

        for recommender_class in collaborative_algorithm_list:

            try:
                runParameterSearch_Collaborative_partial(recommender_class)

            except Exception as e:
                print("On recommender {} Exception {}".format(recommender_class, str(e)))
                traceback.print_exc()


        data_loader = DataIO(folder_path=output_folder_path)
        search_metadata = data_loader.load_data(recommender_class.RECOMMENDER_NAME + "_metadata.zip")
        best_parameters = search_metadata["hyperparameters_best"]  # dictionary with all the fit parameters

        print("best_parameters", best_parameters)

        break

    except (ValueError, IndexError):
        print('Error. Please enter number between 1 and {}'.format(i))




# recommender_list = [
#     # KNN Collaborative filtering
#     'ItemKNNCFRecommender',
#     'UserKNNCFRecommender',
#     'ItemKNNCBFRecommender']
#     # 'itemKNNCF+P3alpha',
#     # 'itemKNNCF+pureSVD',
#     # 'itemKNNCF+TopPop',
#     # 'SLIM_BPR']
#
# print('Recommender Systems: ')
# for i, recomm_type in enumerate(recommender_list, start=1):
#     print('{}. {}'.format(i, recomm_type))

# Fit the model
# -------------

# while True:
#     try:
#         selected = int(input('\nSelect a recommender system: '.format(i)))
#         recomm_type = recommender_list[selected - 1]
#         print('\n ... {} ... '.format(recomm_type))
#
#
#         if recomm_type == 'ItemKNNCBFRecommender':
#
#             #               from utils.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
#             recommender_class = ItemKNNCBFRecommender
#             best_parameters, URM_train, URM_test = parameter_tuning(URM, recommender_class=recommender_class)
#
#             # Use an ItemKNNCF with the parameters we just learned
#             recommender = ItemKNNCBFRecommender(URM_train)
#             recommender.fit(**best_parameters)
#
#             # Evaluate recommender
#             evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])
#             result_dict, _ = evaluator_test.evaluateRecommender(recommender)
#
#             print("result_dict MAP", result_dict[10]["MAP"])
#
#             predictions = input('\nCompute and save top10 predictions?: '
#                                 'y - Yes  n - No\n')
#
#             if predictions == 'y':
#                 # Train the model on the whole dataset using tuned params
#                 recommender = ItemKNNCFRecommender(URM)
#                 recommender.fit(**best_parameters)
#
#                 top_10_items = {}
#                 target_user_id_list = get_target_users()
#
#                 for user_id in target_user_id_list:
#                     item_list = ''
#                     for item in range(10):  # recommended_items
#                         item_list = recommender.recommend(user_id, cutoff=10)
#                         item_list = np.array(item_list)  # list to np.array
#
#                         top_10_items[user_id] = item_list  # .strip() # remove trailing space
#
#                 # save predictions on csv file
#                 create_csv(top_10_items, recomm_type)
#
#
#         break
#
#     except (ValueError, IndexError):
#         print('Error. Please enter number between 1 and {}'.format(i))