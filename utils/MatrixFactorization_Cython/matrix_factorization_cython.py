#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Cython is a superset of Python, allowing you to use C-like operations and import C code.
"""

# Load Cython magic command, this takes care of the compilation step.
# If you are writing code outside Jupyter you'll have to compile using other tools


# ------------------------------    Jupyter Notebook    ---------------------------------------------------------- #


####### Jupyter Notebook #######
# ---------------------- #######
# takes care of the compilation step
# %load_ext Cython or %reload_ext Cython


#%% cython

# BaseRecommender.py
# ------------------------------------------------------------------

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Maurizio Ferrari Dacrema
"""

import numpy as np

# from utils.compute_similarity import check_matrix

class BaseRecommender(object):
    """Abstract BaseRecommender"""

    RECOMMENDER_NAME = "Recommender_Base_Class"

    def __init__(self, URM_train, verbose=True):

        super(BaseRecommender, self).__init__()

        self.URM_train = check_matrix(URM_train.copy(), 'csr', dtype=np.float32)
        self.URM_train.eliminate_zeros()

        self.n_users, self.n_items = self.URM_train.shape
        self.verbose = verbose

        self.filterTopPop = False
        self.filterTopPop_ItemsID = np.array([], dtype=np.int)

        self.items_to_ignore_flag = False
        self.items_to_ignore_ID = np.array([], dtype=np.int)

        self._cold_user_mask = np.ediff1d(self.URM_train.indptr) == 0

        if self._cold_user_mask.any():
            self._print("URM Detected {} ({:.2f} %) cold users.".format(
                self._cold_user_mask.sum(), self._cold_user_mask.sum() / self.n_users * 100))

        self._cold_item_mask = np.ediff1d(self.URM_train.tocsc().indptr) == 0

        if self._cold_item_mask.any():
            self._print("URM Detected {} ({:.2f} %) cold items.".format(
                self._cold_item_mask.sum(), self._cold_item_mask.sum() / self.n_items * 100))

    def _get_cold_user_mask(self):
        return self._cold_user_mask

    def _get_cold_item_mask(self):
        return self._cold_item_mask

    def _print(self, string):
        if self.verbose:
            print("{}: {}".format(self.RECOMMENDER_NAME, string))

    def fit(self):
        pass

    def get_URM_train(self):
        return self.URM_train.copy()

    def set_items_to_ignore(self, items_to_ignore):
        self.items_to_ignore_flag = True
        self.items_to_ignore_ID = np.array(items_to_ignore, dtype=np.int)

    def reset_items_to_ignore(self):
        self.items_to_ignore_flag = False
        self.items_to_ignore_ID = np.array([], dtype=np.int)

    #########################################################################################################
    ##########                                                                                     ##########
    ##########                     COMPUTE AND FILTER RECOMMENDATION LIST                          ##########
    ##########                                                                                     ##########
    #########################################################################################################

    def _remove_TopPop_on_scores(self, scores_batch):
        scores_batch[:, self.filterTopPop_ItemsID] = -np.inf
        return scores_batch

    def _remove_custom_items_on_scores(self, scores_batch):
        scores_batch[:, self.items_to_ignore_ID] = -np.inf
        return scores_batch

    def _remove_seen_on_scores(self, user_id, scores):

        assert self.URM_train.getformat() == "csr", "Recommender_Base_Class: URM_train is not CSR, this will cause errors in filtering seen items"

        seen = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]

        scores[seen] = -np.inf
        return scores

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        :param user_id_array:       array containing the user indices whose recommendations need to be computed
        :param items_to_compute:    array containing the items whose scores are to be computed.
                                        If None, all items are computed, otherwise discarded items will have as score -np.inf
        :return:                    array (len(user_id_array), n_items) with the score.
        """
        raise NotImplementedError(
            "BaseRecommender: compute_item_score not assigned for current recommender, unable to compute prediction scores")

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_custom_items_flag=False, return_scores=False):

        # If is a scalar transform it in a 1-cell array
        if np.isscalar(user_id_array):
            user_id_array = np.atleast_1d(user_id_array)
            single_user = True
        else:
            single_user = False

        if cutoff is None:
            cutoff = self.URM_train.shape[1] - 1

        # Compute the scores using the model-specific function
        # Vectorize over all users in user_id_array
        scores_batch = self._compute_item_score(user_id_array, items_to_compute=items_to_compute)

        for user_index in range(len(user_id_array)):

            user_id = user_id_array[user_index]

            if remove_seen_flag:
                scores_batch[user_index, :] = self._remove_seen_on_scores(user_id, scores_batch[user_index, :])

            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # - Partition the data to extract the set of relevant items
            # - Sort only the relevant items
            # - Get the original item index
            # relevant_items_partition = (-scores_user).argpartition(cutoff)[0:cutoff]
            # relevant_items_partition_sorting = np.argsort(-scores_user[relevant_items_partition])
            # ranking = relevant_items_partition[relevant_items_partition_sorting]
            #
            # ranking_list.append(ranking)

        if remove_top_pop_flag:
            scores_batch = self._remove_TopPop_on_scores(scores_batch)

        if remove_custom_items_flag:
            scores_batch = self._remove_custom_items_on_scores(scores_batch)

        # relevant_items_partition is block_size x cutoff
        relevant_items_partition = (-scores_batch).argpartition(cutoff, axis=1)[:, 0:cutoff]

        # Get original value and sort it
        # [:, None] adds 1 dimension to the array, from (block_size,) to (block_size,1)
        # This is done to correctly get scores_batch value as [row, relevant_items_partition[row,:]]
        relevant_items_partition_original_value = scores_batch[
            np.arange(scores_batch.shape[0])[:, None], relevant_items_partition]
        relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value, axis=1)
        ranking = relevant_items_partition[
            np.arange(relevant_items_partition.shape[0])[:, None], relevant_items_partition_sorting]

        ranking_list = [None] * ranking.shape[0]

        # Remove from the recommendation list any item that has a -inf score
        # Since -inf is a flag to indicate an item to remove
        for user_index in range(len(user_id_array)):
            user_recommendation_list = ranking[user_index]
            user_item_scores = scores_batch[user_index, user_recommendation_list]

            not_inf_scores_mask = np.logical_not(np.isinf(user_item_scores))

            user_recommendation_list = user_recommendation_list[not_inf_scores_mask]
            ranking_list[user_index] = user_recommendation_list.tolist()

        # Return single list for one user, instead of list of lists
        if single_user:
            ranking_list = ranking_list[0]

        if return_scores:
            return ranking_list, scores_batch

        else:
            return ranking_list



# ------------------------------------------------------------------------------------------------------------------ #



# BaseMatrixFactorizationRecommender.py
# ------------------------------------------------------------------

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 16/09/2017
@author: Maurizio Ferrari Dacrema
"""

# from recommenders.BaseRecommender import BaseRecommender
# from KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender

import numpy as np


class BaseMatrixFactorizationRecommender(BaseRecommender):
    """
    This class refers to a BaseRecommender KNN which uses matrix factorization,
    it provides functions to compute item's score as well as a function to save the W_matrix
    The prediction for cold users will always be -inf for ALL items
    """

    def __init__(self, URM_train, verbose=True):
        super(BaseMatrixFactorizationRecommender, self).__init__(URM_train, verbose=verbose)

        self.use_bias = False

    #########################################################################################################
    ##########                                                                                     ##########
    ##########                               COMPUTE ITEM SCORES                                   ##########
    ##########                                                                                     ##########
    #########################################################################################################

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        USER_factors is n_users x n_factors
        ITEM_factors is n_items x n_factors
        The prediction for cold users will always be -inf for ALL items
        :param user_id_array:
        :param items_to_compute:
        :return:
        """

        assert self.USER_factors.shape[1] == self.ITEM_factors.shape[1], \
            "{}: User and Item factors have inconsistent shape".format(self.RECOMMENDER_NAME)

        assert self.USER_factors.shape[0] > np.max(user_id_array), \
            "{}: Cold users not allowed. Users in trained model are {}, requested prediction for users up to {}".format(
                self.RECOMMENDER_NAME, self.USER_factors.shape[0], np.max(user_id_array))

        if items_to_compute is not None:
            item_scores = - np.ones((len(user_id_array), self.ITEM_factors.shape[0]), dtype=np.float32) * np.inf
            item_scores[:, items_to_compute] = np.dot(self.USER_factors[user_id_array],
                                                      self.ITEM_factors[items_to_compute, :].T)

        else:
            item_scores = np.dot(self.USER_factors[user_id_array], self.ITEM_factors.T)

        # No need to select only the specific negative items or warm users because the -inf score will not change
        if self.use_bias:
            item_scores += self.ITEM_bias + self.GLOBAL_bias
            item_scores = (item_scores.T + self.USER_bias[user_id_array]).T

        return item_scores



# ------------------------------------------------------------------------------------------------------------------ #



# MatrixFactorization_Cython_Epoch.py
# ------------------------------------------------------------------



# ------------------------------------------------------------------------------------------------------------------ #



# seconds_to_biggest_unit.py
# ------------------------------------------------------------------

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30/03/2019
@author: Maurizio Ferrari Dacrema
"""


def seconds_to_biggest_unit(time_in_seconds, data_array=None):
    conversion_factor = [
        ("sec", 60),
        ("min", 60),
        ("hour", 24),
        ("day", 365),
    ]

    terminate = False
    unit_index = 0

    new_time_value = time_in_seconds
    new_time_unit = "sec"

    while not terminate:

        next_time = new_time_value / conversion_factor[unit_index][1]

        if next_time >= 1.0:
            new_time_value = next_time

            if data_array is not None:
                data_array /= conversion_factor[unit_index][1]

            unit_index += 1
            new_time_unit = conversion_factor[unit_index][0]

        else:
            terminate = True

    if data_array is not None:
        return new_time_value, new_time_unit, data_array

    else:
        return new_time_value, new_time_unit



# ------------------------------------------------------------------------------------------------------------------ #



# Incremental_Training_Early_Stopping.py
# ------------------------------------------------------------------

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/07/2018
@author: Maurizio Ferrari Dacrema
"""

import time, sys
import numpy as np


# from Base.BaseTempFolder import BaseTempFolder
# from utils.Evaluation.Utils.seconds_to_biggest_unit import seconds_to_biggest_unit


class Incremental_Training_Early_Stopping(object):
    """
    This class provides a function which trains a model applying early stopping
    The term "incremental" refers to the model that is updated at every epoch
    The term "best" refers to the incremental model which corresponded to the best validation score
    The object must implement the following methods:
    _run_epoch(self, num_epoch)                 : trains the model for one epoch (e.g. calling another object implementing the training cython, pyTorch...)
    _prepare_model_for_validation(self)         : ensures the recommender being trained can compute the predictions needed for the validation step
    _update_best_model(self)                    : updates the best model with the current incremental one
    _train_with_early_stopping(.)               : Function that executes the training, validation and early stopping by using the previously implemented functions
    """

    def __init__(self):
        super(Incremental_Training_Early_Stopping, self).__init__()

    def get_early_stopping_final_epochs_dict(self):
        """
        This function returns a dictionary to be used as optimal parameters in the .fit() function
        It provides the flexibility to deal with multiple early-stopping in a single algorithm
        e.g. in NeuMF there are three model components each with its own optimal number of epochs
        the return dict would be {"epochs": epochs_best_neumf, "epochs_gmf": epochs_best_gmf, "epochs_mlp": epochs_best_mlp}
        :return:
        """

        return {"epochs": self.epochs_best}

    def _run_epoch(self, num_epoch):
        """
        This function should run a single epoch on the object you train. This may either involve calling a function to do an epoch
        on a Cython object or a loop on the data points directly in python
        :param num_epoch:
        :return:
        """
        raise NotImplementedError()

    def _prepare_model_for_validation(self):
        """
        This function is executed before the evaluation of the current model
        It should ensure the current object "self" can be passed to the evaluator object
        E.G. if the epoch is done via Cython or PyTorch, this function should get the new parameter values from
        the cython or pytorch objects into the self. pyhon object
        :return:
        """
        raise NotImplementedError()

    def _update_best_model(self):
        """
        This function is called when the incremental model is found to have better validation score than the current best one
        So the current best model should be replaced by the current incremental one.
        Important, remember to clone the objects and NOT to create a pointer-reference, otherwise the best solution will be altered
        by the next epoch
        :return:
        """
        raise NotImplementedError()

    def _train_with_early_stopping(self, epochs_max, epochs_min=0,
                                   validation_every_n=None, stop_on_validation=False,
                                   validation_metric=None, lower_validations_allowed=None, evaluator_object=None,
                                   algorithm_name="Incremental_Training_Early_Stopping"):
        """
        :param epochs_max:                  max number of epochs the training will last
        :param epochs_min:                  min number of epochs the training will last
        :param validation_every_n:          number of epochs after which the model will be evaluated and a best_model selected
        :param stop_on_validation:          [True/False] whether to stop the training before the max number of epochs
        :param validation_metric:           which metric to use when selecting the best model, higher values are better
        :param lower_validations_allowed:    number of contiguous validation steps required for the tranining to early-stop
        :param evaluator_object:            evaluator instance used to compute the validation metrics.
                                                If multiple cutoffs are available, the first one is used
        :param algorithm_name:              name of the algorithm to be displayed in the output updates
        :return: -
        Supported uses:
        - Train for max number of epochs with no validation nor early stopping:
            _train_with_early_stopping(epochs_max = 100,
                                        evaluator_object = None
                                        epochs_min,                 not used
                                        validation_every_n,         not used
                                        stop_on_validation,         not used
                                        validation_metric,          not used
                                        lower_validations_allowed,   not used
                                        )
        - Train for max number of epochs with validation but NOT early stopping:
            _train_with_early_stopping(epochs_max = 100,
                                        evaluator_object = evaluator
                                        stop_on_validation = False
                                        validation_every_n = int value
                                        validation_metric = metric name string
                                        epochs_min,                 not used
                                        lower_validations_allowed,   not used
                                        )
        - Train for max number of epochs with validation AND early stopping:
            _train_with_early_stopping(epochs_max = 100,
                                        epochs_min = int value
                                        evaluator_object = evaluator
                                        stop_on_validation = True
                                        validation_every_n = int value
                                        validation_metric = metric name string
                                        lower_validations_allowed = int value
                                        )
        """

        assert epochs_max >= 0, "{}: Number of epochs_max must be >= 0, passed was {}".format(algorithm_name,
                                                                                              epochs_max)
        assert epochs_min >= 0, "{}: Number of epochs_min must be >= 0, passed was {}".format(algorithm_name,
                                                                                              epochs_min)
        assert epochs_min <= epochs_max, "{}: epochs_min must be <= epochs_max, passed are epochs_min {}, epochs_max {}".format(
            algorithm_name, epochs_min, epochs_max)

        # Train for max number of epochs with no validation nor early stopping
        # OR Train for max number of epochs with validation but NOT early stopping
        # OR Train for max number of epochs with validation AND early stopping
        assert evaluator_object is None or \
               (
                       evaluator_object is not None and not stop_on_validation and validation_every_n is not None and validation_metric is not None) or \
               (
                       evaluator_object is not None and stop_on_validation and validation_every_n is not None and validation_metric is not None and lower_validations_allowed is not None), \
            "{}: Inconsistent parameters passed, please check the supported uses".format(algorithm_name)

        start_time = time.time()

        self.best_validation_metric = None
        lower_validatons_count = 0
        convergence = False

        self.epochs_best = 0

        epochs_current = 0

        while epochs_current < epochs_max and not convergence:

            self._run_epoch(epochs_current)

            # If no validation required, always keep the latest
            if evaluator_object is None:

                self.epochs_best = epochs_current

            # Determine whether a validaton step is required
            elif (epochs_current + 1) % validation_every_n == 0:

                print("{}: Validation begins...".format(algorithm_name))

                self._prepare_model_for_validation()

                # If the evaluator validation has multiple cutoffs, choose the first one
                results_run, results_run_string = evaluator_object.evaluateRecommender(self)
                results_run = results_run[list(results_run.keys())[0]]

                print("{}: {}".format(algorithm_name, results_run_string))

                # Update optimal model
                current_metric_value = results_run[validation_metric]
                #
                # if not np.isfinite(current_metric_value):
                #     if isinstance(self, BaseTempFolder):
                #         # If the recommender uses BaseTempFolder, clean the temp folder
                #         self._clean_temp_folder(temp_file_folder=self.temp_file_folder)
                #
                #     assert False, "{}: metric value is not a finite number, terminating!".format(self.RECOMMENDER_NAME)
                #

                if self.best_validation_metric is None or self.best_validation_metric < current_metric_value:

                    print("{}: New best model found! Updating.".format(algorithm_name))

                    self.best_validation_metric = current_metric_value

                    self._update_best_model()

                    self.epochs_best = epochs_current + 1
                    lower_validatons_count = 0

                else:
                    lower_validatons_count += 1

                if stop_on_validation and lower_validatons_count >= lower_validations_allowed and epochs_current >= epochs_min:
                    convergence = True

                    elapsed_time = time.time() - start_time
                    new_time_value, new_time_unit = seconds_to_biggest_unit(elapsed_time)

                    print(
                        "{}: Convergence reached! Terminating at epoch {}. Best value for '{}' at epoch {} is {:.4f}. Elapsed time {:.2f} {}".format(
                            algorithm_name, epochs_current + 1, validation_metric, self.epochs_best,
                            self.best_validation_metric, new_time_value, new_time_unit))

            elapsed_time = time.time() - start_time
            new_time_value, new_time_unit = seconds_to_biggest_unit(elapsed_time)

            print("{}: Epoch {} of {}. Elapsed time {:.2f} {}".format(
                algorithm_name, epochs_current + 1, epochs_max, new_time_value, new_time_unit))

            epochs_current += 1

            sys.stdout.flush()
            sys.stderr.flush()

        # If no validation required, keep the latest
        if evaluator_object is None:
            self._prepare_model_for_validation()
            self._update_best_model()

        # Stop when max epochs reached and not early-stopping
        if not convergence:
            elapsed_time = time.time() - start_time
            new_time_value, new_time_unit = seconds_to_biggest_unit(elapsed_time)

            if evaluator_object is not None and self.best_validation_metric is not None:
                print(
                    "{}: Terminating at epoch {}. Best value for '{}' at epoch {} is {:.4f}. Elapsed time {:.2f} {}".format(
                        algorithm_name, epochs_current, validation_metric, self.epochs_best,
                        self.best_validation_metric, new_time_value, new_time_unit))
            else:
                print("{}: Terminating at epoch {}. Elapsed time {:.2f} {}".format(
                    algorithm_name, epochs_current, new_time_value, new_time_unit))



# ------------------------------------------------------------------------------------------------------------------ #



# _MatrixFactorization_Cython
# ------------------------------------------------------------------

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07/09/17
@author: Maurizio Ferrari Dacrema
"""

# from Base.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
# from Base.Incremental_Training_Early_Stopping import Incremental_Training_Early_Stopping
# from Base.Recommender_utils import check_matrix

# from CythonCompiler.run_compile_subprocess import run_compile_subprocess
import os, sys
import numpy as np


class _MatrixFactorization_Cython(BaseMatrixFactorizationRecommender, Incremental_Training_Early_Stopping):
    RECOMMENDER_NAME = "MatrixFactorization_Cython_Recommender"

    def __init__(self, URM_train, verbose=True, recompile_cython=False, algorithm_name="MF_BPR"):
        super(_MatrixFactorization_Cython, self).__init__(URM_train, verbose=verbose)

        self.n_users, self.n_items = self.URM_train.shape
        self.normalize = False
        self.algorithm_name = algorithm_name

        if recompile_cython:
            print("Compiling in Cython")
            self.runCompilationScript()
            print("Compilation Complete")

    def fit(self, epochs=300, batch_size=1000,
            num_factors=10, positive_threshold_BPR=None,
            learning_rate=0.001, use_bias=True,
            sgd_mode='sgd',
            negative_interactions_quota=0.0,
            init_mean=0.0, init_std_dev=0.1,
            user_reg=0.0, item_reg=0.0, bias_reg=0.0, positive_reg=0.0, negative_reg=0.0,
            random_seed=None,
            **earlystopping_kwargs):

        self.num_factors = num_factors
        self.use_bias = use_bias
        self.sgd_mode = sgd_mode
        self.positive_threshold_BPR = positive_threshold_BPR
        self.learning_rate = learning_rate

        assert negative_interactions_quota >= 0.0 and negative_interactions_quota < 1.0, "{}: negative_interactions_quota must be a float value >=0 and < 1.0, provided was '{}'".format(
            self.RECOMMENDER_NAME, negative_interactions_quota)
        self.negative_interactions_quota = negative_interactions_quota

        # Import compiled module
        #         from MatrixFactorization.Cython.MatrixFactorization_Cython_Epoch import MatrixFactorization_Cython_Epoch

        if self.algorithm_name in ["FUNK_SVD", "ASY_SVD"]:

            self.cythonEpoch = MatrixFactorization_Cython_Epoch(self.URM_train,
                                                                algorithm_name=self.algorithm_name,
                                                                n_factors=self.num_factors,
                                                                learning_rate=learning_rate,
                                                                sgd_mode=sgd_mode,
                                                                user_reg=user_reg,
                                                                item_reg=item_reg,
                                                                bias_reg=bias_reg,
                                                                batch_size=batch_size,
                                                                use_bias=use_bias,
                                                                init_mean=init_mean,
                                                                negative_interactions_quota=negative_interactions_quota,
                                                                init_std_dev=init_std_dev,
                                                                verbose=self.verbose,
                                                                random_seed=random_seed)

        elif self.algorithm_name == "MF_BPR":

            # Select only positive interactions
            URM_train_positive = self.URM_train.copy()

            if self.positive_threshold_BPR is not None:
                URM_train_positive.data = URM_train_positive.data >= self.positive_threshold_BPR
                URM_train_positive.eliminate_zeros()

                assert URM_train_positive.nnz > 0, "MatrixFactorization_Cython: URM_train_positive is empty, positive threshold is too high"

            self.cythonEpoch = MatrixFactorization_Cython_Epoch(URM_train_positive,
                                                                algorithm_name=self.algorithm_name,
                                                                n_factors=self.num_factors,
                                                                learning_rate=learning_rate,
                                                                sgd_mode=sgd_mode,
                                                                user_reg=user_reg,
                                                                positive_reg=positive_reg,
                                                                negative_reg=negative_reg,
                                                                batch_size=batch_size,
                                                                use_bias=use_bias,
                                                                init_mean=init_mean,
                                                                init_std_dev=init_std_dev,
                                                                verbose=self.verbose,
                                                                random_seed=random_seed)
        self._prepare_model_for_validation()
        self._update_best_model()

        self._train_with_early_stopping(epochs,
                                        algorithm_name=self.algorithm_name,
                                        **earlystopping_kwargs)

        self.USER_factors = self.USER_factors_best
        self.ITEM_factors = self.ITEM_factors_best

        if self.use_bias:
            self.USER_bias = self.USER_bias_best
            self.ITEM_bias = self.ITEM_bias_best
            self.GLOBAL_bias = self.GLOBAL_bias_best

        sys.stdout.flush()

    def _prepare_model_for_validation(self):
        self.USER_factors = self.cythonEpoch.get_USER_factors()
        self.ITEM_factors = self.cythonEpoch.get_ITEM_factors()

        if self.use_bias:
            self.USER_bias = self.cythonEpoch.get_USER_bias()
            self.ITEM_bias = self.cythonEpoch.get_ITEM_bias()
            self.GLOBAL_bias = self.cythonEpoch.get_GLOBAL_bias()

    def _update_best_model(self):
        self.USER_factors_best = self.USER_factors.copy()
        self.ITEM_factors_best = self.ITEM_factors.copy()

        if self.use_bias:
            self.USER_bias_best = self.USER_bias.copy()
            self.ITEM_bias_best = self.ITEM_bias.copy()
            self.GLOBAL_bias_best = self.GLOBAL_bias

    def _run_epoch(self, num_epoch):
        self.cythonEpoch.epochIteration_Cython()

    def runCompilationScript(self):

        # Run compile script setting the working directory to ensure the compiled file are contained in the
        # appropriate subfolder and not the project root

        file_subfolder = "/MatrixFactorization/Cython"
        file_to_compile_list = ['MatrixFactorization_Cython_Epoch.pyx']

        run_compile_subprocess(file_subfolder, file_to_compile_list)

        print("{}: Compiled module {} in subfolder: {}".format(self.RECOMMENDER_NAME, file_to_compile_list,
                                                               file_subfolder))

        # Command to run compilation script
        # python compile_script.py MatrixFactorization_Cython_Epoch.pyx build_ext --inplace

        # Command to generate html report
        # cython -a MatrixFactorization_Cython_Epoch.pyx


class MatrixFactorization_BPR_Cython(_MatrixFactorization_Cython):
    """
    Subclas allowing only for MF BPR
    """

    RECOMMENDER_NAME = "MatrixFactorization_BPR_Cython_Recommender"

    def __init__(self, *pos_args, **key_args):
        super(MatrixFactorization_BPR_Cython, self).__init__(*pos_args, algorithm_name="MF_BPR", **key_args)

    def fit(self, **key_args):
        key_args["use_bias"] = False
        key_args["negative_interactions_quota"] = 0.0

        super(MatrixFactorization_BPR_Cython, self).fit(**key_args)


class MatrixFactorization_FunkSVD_Cython(_MatrixFactorization_Cython):
    """
    Subclas allowing only for FunkSVD model
    Reference: http://sifter.org/~simon/journal/20061211.html
    Factorizes the rating matrix R into the dot product of two matrices U and V of latent factors.
    U represent the user latent factors, V the item latent factors.
    The model is learned by solving the following regularized Least-squares objective function with Stochastic Gradient Descent
    \operatornamewithlimits{argmin} \limits_{U,V}\frac{1}{2}||R - UV^T||^2_2 + \frac{\lambda}{2}(||U||^2_F + ||V||^2_F)
    Latent factors are initialized from a Normal distribution with given mean and std.
    """

    RECOMMENDER_NAME = "MatrixFactorization_FunkSVD_Cython_Recommender"

    def __init__(self, *pos_args, **key_args):
        super(MatrixFactorization_FunkSVD_Cython, self).__init__(*pos_args, algorithm_name="FUNK_SVD", **key_args)

    def fit(self, **key_args):
        super(MatrixFactorization_FunkSVD_Cython, self).fit(**key_args)


class MatrixFactorization_AsySVD_Cython(_MatrixFactorization_Cython):
    """
    Subclas allowing only for AsymmetricSVD model
    Reference: Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model (Koren, 2008)
    Factorizes the rating matrix R into two matrices X and Y of latent factors, which both represent item latent features.
    Users are represented by aggregating the latent features in Y of items they have already rated.
    Rating prediction is performed by computing the dot product of this accumulated user profile with the target item's
    latent factor in X.
    The model is learned by solving the following regularized Least-squares objective function with Stochastic Gradient Descent
    \operatornamewithlimits{argmin}\limits_{x*,y*}\frac{1}{2}\sum_{i,j \in R}(r_{ij} - x_j^T \sum_{l \in R(i)} r_{il}y_l)^2 + \frac{\lambda}{2}(\sum_{i}{||x_i||^2} + \sum_{j}{||y_j||^2})
    """

    RECOMMENDER_NAME = "MatrixFactorization_AsySVD_Cython_Recommender"

    def __init__(self, *pos_args, **key_args):
        super(MatrixFactorization_AsySVD_Cython, self).__init__(*pos_args, algorithm_name="ASY_SVD", **key_args)

    def fit(self, **key_args):

        if "batch_size" in key_args and key_args["batch_size"] > 1:
            print("{}: batch_size not supported for this recommender, setting to default value 1.".format(
                self.RECOMMENDER_NAME))

        key_args["batch_size"] = 1

        super(MatrixFactorization_AsySVD_Cython, self).fit(**key_args)

    def _prepare_model_for_validation(self):
        """
        AsymmetricSVD Computes two |n_items| x |n_features| matrices of latent factors
        ITEM_factors_Y must be used to estimate user's latent factors via the items they interacted with
        :return:
        """

        self.ITEM_factors_Y = self.cythonEpoch.get_USER_factors()
        self.USER_factors = self._estimate_user_factors(self.ITEM_factors_Y)

        self.ITEM_factors = self.cythonEpoch.get_ITEM_factors()

        if self.use_bias:
            self.USER_bias = self.cythonEpoch.get_USER_bias()
            self.ITEM_bias = self.cythonEpoch.get_ITEM_bias()
            self.GLOBAL_bias = self.cythonEpoch.get_GLOBAL_bias()

    def _update_best_model(self):
        self.USER_factors_best = self.USER_factors.copy()
        self.ITEM_factors_best = self.ITEM_factors.copy()
        self.ITEM_factors_Y_best = self.ITEM_factors_Y.copy()

        if self.use_bias:
            self.USER_bias_best = self.USER_bias.copy()
            self.ITEM_bias_best = self.ITEM_bias.copy()
            self.GLOBAL_bias_best = self.GLOBAL_bias

    def _estimate_user_factors(self, ITEM_factors_Y):

        profile_length = np.ediff1d(self.URM_train.indptr)
        profile_length_sqrt = np.sqrt(profile_length)

        # Estimating the USER_factors using ITEM_factors_Y
        if self.verbose:
            print("{}: Estimating user factors... ".format(self.algorithm_name))

        USER_factors = self.URM_train.dot(ITEM_factors_Y)

        # Divide every row for the sqrt of the profile length
        for user_index in range(self.n_users):

            if profile_length_sqrt[user_index] > 0:
                USER_factors[user_index, :] /= profile_length_sqrt[user_index]

        if self.verbose:
            print("{}: Estimating user factors... done!".format(self.algorithm_name))

        return USER_factors



# ------------------------------------------------------------------------------------------------------------------ #



# check_matrix
# ------------------------------------------------------------------

# Function hiding some conversion checks
def check_matrix(X, format='csc', dtype=np.float32):
    if format == 'csc' and not isinstance(X, sps.csc_matrix):
        return X.tocsc().astype(dtype)
    elif format == 'csr' and not isinstance(X, sps.csr_matrix):
        return X.tocsr().astype(dtype)
    elif format == 'coo' and not isinstance(X, sps.coo_matrix):
        return X.tocoo().astype(dtype)
    elif format == 'dok' and not isinstance(X, sps.dok_matrix):
        return X.todok().astype(dtype)
    elif format == 'bsr' and not isinstance(X, sps.bsr_matrix):
        return X.tobsr().astype(dtype)
    elif format == 'dia' and not isinstance(X, sps.dia_matrix):
        return X.todia().astype(dtype)
    elif format == 'lil' and not isinstance(X, sps.lil_matrix):
        return X.tolil().astype(dtype)
    else:
        return X.astype(dtype)



# ------------------------------------------------------------------------------------------------------------------ #



# run_compile_subprocess.py
# ------------------------------------------------------------------

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 06/01/2018
@author: Maurizio Ferrari Dacrema
"""

import subprocess, os, sys, shutil


def run_compile_subprocess(file_subfolder, file_to_compile_list):
    # Run compile script setting the working directory to ensure the compiled file are contained in the
    # appropriate subfolder and not the project root

    current_python_path = sys.executable

    compile_script_absolute_path = os.getcwd() + '/CythonCompiler/compile_script.py'
    file_subfolder_absolute_path = os.getcwd() + "/" + file_subfolder

    for file_to_compile in file_to_compile_list:

        try:
            command = [current_python_path,
                       compile_script_absolute_path,
                       file_to_compile,
                       'build_ext',
                       '--inplace'
                       ]

            output = subprocess.check_output(' '.join(command),
                                             shell=True,
                                             cwd=file_subfolder_absolute_path)

            try:

                command = ['cython',
                           file_to_compile,
                           '-a'
                           ]
                output = subprocess.check_output(' '.join(command),
                                                 shell=True,
                                                 cwd=file_subfolder_absolute_path)
            except:
                pass

        except Exception as exc:
            raise exc

        finally:
            # Removing temporary "build" subfolder
            shutil.rmtree(file_subfolder_absolute_path + "/build", ignore_errors=True)

    # Command to run compilation script
    # python CythonCompiler/compile_script.py filename.pyx build_ext --inplace

    # Command to generate html report
    # cython -a filename.pyx

