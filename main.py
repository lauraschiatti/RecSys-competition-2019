#!/usr/bin/env python3
#  -*- coding: utf-8 -*-


import traceback
import numpy as np
from utils.data_manager import build_URM, build_ICM, get_target_users
from utils.Evaluation.Evaluator import EvaluatorHoldout
from utils.ParameterTuning.hyperparameter_search import runParameterSearch_Collaborative, runParameterSearch_Content
from utils.DataIO import DataIO
from utils.create_submission_file import create_csv
from utils.data_splitter import split_train_leave_k_out_user_wise
from utils.data_splitter import split_train_validation_random_holdout
from utils import masks

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
from utils.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender


# Build URM, ICM and UCM
# ----------------------

URM_all = build_URM()
ICM_all = build_ICM()
# data_manager.get_statistics_URM(URM)

# Cold items, cold users and cold features
# URM, ICM = masks.refactor_URM_ICM(URM, ICM)

# Top-10 recommenders
at = 10 # k recommended_items


# URM train/validation/test splitting
# -----------------------------------

# from Data_manager.Movielens1M.Movielens1MReader import Movielens1MReader
# from Data_manager.DataSplitter_k_fold_stratified import DataSplitter_Warm_k_fold
# todo: try splitting using get_holdout_split
# dataset_object = Movielens1MReader()
# dataSplitter = DataSplitter_Warm_k_fold(dataset_object)
# dataSplitter.load_data()
# URM_train, URM_validation, URM_test = dataSplitter.get_holdout_split()

# URM_train, URM_test = split_train_validation_random_holdout(URM_all, train_split=0.8)
# URM_train, URM_validation = split_train_validation_random_holdout(URM_train, train_split=0.9)

URM_train, URM_test = split_train_leave_k_out_user_wise(URM_all, k_out = 1, use_validation_set = False, leave_random_out = True)
URM_train, URM_validation = split_train_leave_k_out_user_wise(URM_all, k_out = 1, use_validation_set = False, leave_random_out = True)

# Recommenders
# ------------

# Collaborative recommenders
collaborative_algorithm_list = [
    # Random,
    # TopPop,
    # P3alphaRecommender,
    # RP3betaRecommender,
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
    # ItemKNNCBFRecommender +

]

recommender_list = [
    # Non-personalized
    # Random,
    # TopPop,

    # Collaborative recommenders
    # P3alphaRecommender,
    # RP3betaRecommender,
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
]

# from Utils.PoolWithSubprocess import PoolWithSubprocess
# import multiprocessing
#
# pool = PoolWithSubprocess(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
# resultList = pool.map(runParameterSearch_Collaborative_partial, collaborative_algorithm_list)
# pool.close()
# pool.join()


print('\nRecommender Systems: ')
for i, recomm_type in enumerate(recommender_list, start=1):
    print('{}. {}'.format(i, recomm_type.RECOMMENDER_NAME))

while True:
    try:
        selected = int(input('\nSelect a recommender system: '.format(i)))
        recommender_class = recommender_list[selected - 1]
        print('\n ... {} ... '.format(recommender_class.RECOMMENDER_NAME))


        # Hyperparameters tuning
        # ----------------------

        apply_hyperparams_tuning = True

        if apply_hyperparams_tuning:

            metric_to_optimize = "MAP"

            evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[at])
            evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[at, at+5])
            evaluator_validation_earlystopping = None  # EvaluatorHoldout(URM_train, cutoff_list=[5], exclude_seen = False)
            output_folder_path = "result_experiments/"

            n_cases = 8  # 2
            n_random_starts = 5  # 1

            save_model = "no"
            allow_weighting = True  # provides better results
            similarity_type_list = ["cosine"]

            ICM_name = "ICM_all"

            output_file_name_root = "{}_metadata.zip".format(recommender_class.RECOMMENDER_NAME)

            if recommender_class in collaborative_algorithm_list:
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


            if recommender_class in content_algorithm_list:
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


        # Fit the recommender with the parameters we just learned
        if recommender_class in content_algorithm_list:
            recommender = recommender_class(URM_train, ICM_all) # todo: ICM_all or ICM_train?
        else:
            recommender = recommender_class(URM_train)

        recommender.fit(**best_parameters)

        # Evaluate model
        evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[at])
        result_dict, _ = evaluator_test.evaluateRecommender(recommender)

        print("{} result_dict MAP {}".format(recommender_class.RECOMMENDER_NAME, result_dict[at]["MAP"]))

        # Generate predictions
        predictions = input('\nCompute and save top10 predictions?: y - Yes  n - No\n')

        if predictions == 'y':

            # Train the model on the whole dataset using tuned params
            if recommender_class in content_algorithm_list:
                recommender = recommender_class(URM_all, ICM_all)
            else:
                recommender = recommender_class(URM_all)

            recommender.fit(**best_parameters)

            top_10_items = {}
            target_user_id_list = get_target_users()

            for user_id in target_user_id_list:
                item_list = ''

                for item in range(at):
                    item_list = recommender.recommend(user_id, cutoff=at)
                    item_list = np.array(item_list)  # list to np.array
                    top_10_items[user_id] = item_list

            # save predictions on csv file
            create_csv(top_10_items, recommender_class.RECOMMENDER_NAME)

        break

    except (ValueError, IndexError):
        print('Error. Please enter number between 1 and {}'.format(i))
