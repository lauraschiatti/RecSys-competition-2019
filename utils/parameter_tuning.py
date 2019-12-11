


from utils import data_splitter
from utils.Evaluation.Evaluator import EvaluatorHoldout

# Automate Hyperparameter Optimization using Bayesian Optimization With scikit-optimize
# its purpose is to provide a very simple way to tune some of the most common parameters
# ---------------------------------------------------------------------------------------
def parameter_tuning(URM, recommender_class):

    URM_train, URM_test = data_splitter.split_train_validation_random_holdout(URM, train_split=0.8)
    URM_train, URM_validation = data_splitter.split_train_validation_random_holdout(URM_train, train_split=0.9)

    # Step 1: Import the evaluator objects

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10, 15])


    # Step 2: Create BayesianSearch object

    from utils.ParameterTuning.SearchBayesianSkopt import SearchBayesianSkopt

    parameterSearch = SearchBayesianSkopt(recommender_class,
                                          evaluator_validation=evaluator_validation,
                                          evaluator_test=evaluator_test)

    # Step 3: Define parameters range

    from utils.ParameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
    from skopt.space import Real, Integer, Categorical

    hyperparameters_range_dictionary = {}
    hyperparameters_range_dictionary["topK"] = Integer(5, 1000)
    hyperparameters_range_dictionary["shrink"] = Integer(0, 1000)
    hyperparameters_range_dictionary["similarity"] = Categorical(["cosine"])
    hyperparameters_range_dictionary["normalize"] = Categorical([True, False])

    recommender_input_args = SearchInputRecommenderArgs(
        CONSTRUCTOR_POSITIONAL_ARGS=[URM_train],
        CONSTRUCTOR_KEYWORD_ARGS={},
        FIT_POSITIONAL_ARGS=[],
        FIT_KEYWORD_ARGS={}
    )

    output_folder_path = "result_experiments/"

    import os

    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Step 4: run

    n_cases = 2
    metric_to_optimize = "MAP"

    parameterSearch.search(recommender_input_args,
                           parameter_search_space=hyperparameters_range_dictionary,
                           n_cases=n_cases,
                           n_random_starts=1,
                           save_model="no",
                           output_folder_path=output_folder_path,
                           output_file_name_root=recommender_class.RECOMMENDER_NAME,
                           metric_to_optimize=metric_to_optimize
                           )

    # Step 5: return best_parameters

    from utils.DataIO import DataIO

    data_loader = DataIO(folder_path=output_folder_path)
    search_metadata = data_loader.load_data(recommender_class.RECOMMENDER_NAME + "_metadata.zip")

    print("search_metadata", search_metadata)

    best_parameters = search_metadata["hyperparameters_best"]
    print("best_parameters", best_parameters)

    return best_parameters, URM_train, URM_test