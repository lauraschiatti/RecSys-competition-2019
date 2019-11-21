#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

"""
	create_submission_file.py: module for creating the formatted csv submission file.
"""

import os
from datetime import datetime

def create_csv(top_10_items, recommender):
	print("\nGenerating submission csv ... ")

	# save on a different dir according to the recommender used
	submissions_dir = './submissions/' + recommender

	# If directory for the recommender does not exist, create
	if not os.path.exists(submissions_dir):
		os.makedirs(submissions_dir)

	csv_fname = 'submission_'
	csv_fname += datetime.now().strftime('%b%d_%H-%M-%S')+ '.csv'

	file_path = os.path.join(submissions_dir, csv_fname)

	with open(file_path, 'w') as f:

		fieldnames = 'user_id,item_list'
		f.write(fieldnames + '\n')

		for user_id, item_list in top_10_items.items():
			row = str(user_id) + ',' + str(item_list) + '\n'
			f.write(row.replace('[', '').replace(']', ''))  # remove '[' ']' from item_list string
