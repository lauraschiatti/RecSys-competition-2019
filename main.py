#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

# import scipy.sparse as sps
# import numpy as np

from utils import data_manager as data
from utils import evaluation as eval
from utils import create_submission_file as create_csv

from recommenders import RandomRecommender, TopPopRecommender, GlobalEffectsRecommender


# Build URM
# ---------

URM = data.build_URM()

data.get_statistics_URM()

# todo: deal with both cold items and cold users


# Split into train and test set # todo: local train_test_holdout should be LOO too?
# -------------------------

train_split = 0.8
URM_train, URM_test = data.train_test_holdout(URM, train_split)

# NOTE: if the test data contains a lot of low rating interactions,
	# those interactions are penalized by GlobalEffects
URM_test_positive_only = data.URM_test_positive_only(URM_test)

# URM_train, URM_test =  data.get_URM_train_for_test_fold(URM, n_test_fold=1)

# Train model without left-out ratings)
# ------------------------------------

recommender_list = ['RandomRecommender', 'TopPopRecommender', 'GlobalEffectsRecommender']

print('Recommender Systems: ')
for i, recomm_type in enumerate(recommender_list, start=1):
	print('{}. {}'.format(i, recomm_type))

while True:
	try:
		selected = int(input('\nSelect a recommender system: '.format(i)))
		recomm_type = recommender_list[selected - 1]
		print('\n ... {} ... '.format(recomm_type))

		# fit model
		if recomm_type == 'RandomRecommender':
			recommender = RandomRecommender.RandomRecommender()
			recommender.fit(URM_train)

		elif recomm_type == 'TopPopRecommender':
			recommender = TopPopRecommender.TopPopRecommender()
			recommender.fit(URM_train)


		elif recomm_type == 'GlobalEffectsRecommender':
			recommender = GlobalEffectsRecommender.GlobalEffectsRecommender()
			recommender.fit(URM_train)

		break

	except (ValueError, IndexError):
		print('Error. Please enter number between 1 and {}'.format(i))



# Evaluate model on left-out ratings (URM_test)
# ---------------------------------------------

# positive_only = False
#
# if positive_only:
eval.evaluate_algorithm(URM_test, recommender)

# else:
# 	print('\nevaluation of {} with URM_test_positive_only: '.format(recomm_type))
# 	eval.evaluate_algorithm(URM_test_positive_only, recommender)



# Compute top-10 recommendations for each target user
# ---------------------------------------------------

predictions = input('\nCompute and save top10 predictions ?:'
				 '1 - Yes' 
                 '2 - No')


if predictions == '1':

	top_10_items = {}
	target_user_id_list = data.get_target_users()

	for user_id in target_user_id_list:  # target users

		item_list = ''
		for item in range(10):  # recommended_items
			item_list = recommender.recommend(user_id)

			top_10_items[user_id] = item_list  # .strip() # remove trailing space

	# Prints the nicely formatted dictionary
	# import pprint
	# pprint.pprint(top_10_items)

	# save predictions on csv file
	create_csv.create_csv(top_10_items, recomm_type)
