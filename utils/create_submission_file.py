
import os
from datetime import datetime

def create_csv(top_10_items, recommender, results_dir='./submissions'):

	csv_fname = 'submission_'
	csv_fname += datetime.now().strftime('%b%d_%H-%M-%S')+ '.csv'

	with open(os.path.join(results_dir, csv_fname), 'w') as f:

		fieldnames = 'user_id,item_list'
		f.write(fieldnames + '\n')

		for user_id, item_list in top_10_items.items():
			f.write(str(user_id) + ',' + str(item_list) + '\n')


top_10_items = {}
for user_id in range(100): # users
	item_list = ''
	for item in range(10): # 10 relevant items
		item_list += str(item) + ' '

		top_10_items[user_id] = item_list.strip() # remove trailing space


# @Todo: save on a different folder according to the recommender used for the predictions
create_csv(top_10_items, '')