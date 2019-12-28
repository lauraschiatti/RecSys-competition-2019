#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

"""
	create_submission_file.py: module for creating the formatted csv submission file.
"""

import os
from datetime import datetime
import numpy as np


def create_csv(user_id_array, item_list, recommender_name):
    print("\nGenerating submission csv ... ")

    # save on a different dir according to the recommender used
    if recommender_name != None:
        submissions_dir = './submissions/' + recommender_name
    else:
        submissions_dir = './submissions/'

    # If directory for the recommender does not exist, create
    if not os.path.exists(submissions_dir):
        os.makedirs(submissions_dir)

    csv_fname = 'submission_' + datetime.now().strftime('%b%d_%H-%M-%S') + '.csv'
    csv_file = os.path.join(submissions_dir, csv_fname)

    # field names
    header = ['user_id', 'item_list']

    # data rows of csv file
    items_by_user = list(zip(user_id_array, item_list))  # zip list

    with open(csv_file, 'w', newline='') as file:
        fieldnames = 'user_id,item_list'
        file.write(fieldnames + '\n')

        for item_list in items_by_user:
            row = str(item_list[0]) + ',' + str(np.array(item_list[1])) + '\n'
            file.write(row.replace('[', '').replace(']', ''))  # remove '[' ']' from item_list string
