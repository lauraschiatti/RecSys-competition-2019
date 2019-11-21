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

data.interactions_statistics()


# Train/test data splitting # todo: local train_test_holdout should be LOO too?
# -------------------------

train_perc = 0.8
URM_train, URM_test = data.train_test_holdout(URM, train_perc)


# Train model without left-out ratings)
# ------------------------------------

recommender_list = ['RandomRecommender', 'TopPopRecommender', 'GlobalEffectsRecommender']

print("Recommender Systems: ")
for i, recomm_type in enumerate(recommender_list, start=1):
	print('{}. {}'.format(i, recomm_type))


while True:
	try:
		selected = int(input('\nSelect a recommender system: '.format(i)))
		recomm_type = recommender_list[selected - 1]
		print('\n ... {} ... '.format(recomm_type))

		# fit model
		if recomm_type == "RandomRecommender":
			recommender = RandomRecommender.RandomRecommender()
			recommender.fit(URM_train)

		# todo: check TopPop and GlobalEffects
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

eval.evaluate_algorithm(URM_test, recommender)


# Compute top-10 recommendations for each target user
# ---------------------------------------------------

top_10_items = {}
target_user_id_list = data.get_target_users()

for user_id in target_user_id_list:  # target users

	item_list = ''
	for item in range(10):  # recommended_items
		item_list = recommender.recommend(user_id)

		top_10_items[user_id] = item_list #.strip() # remove trailing space


# Prints the nicely formatted dictionary
# import pprint
# pprint.pprint(top_10_items)

create_csv.create_csv(top_10_items, recomm_type)
