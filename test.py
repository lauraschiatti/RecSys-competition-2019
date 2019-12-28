#!/usr/bin/env python3
#  -*- coding: utf-8 -*-



# # The recommendation quality of the three algorithms changes depending on the user profile length
# profile_length = np.ediff1d(URM_train.indptr)
#
# # Let's select a few groups of 10% of the users with the least number of interactions
# block_size = int(len(profile_length) * 0.10)
# # print("block_size", block_size)
#
# n_users, n_items = URM_all.shape #todo: URM_train
# # print("n_users {}, n_items {}".format(n_users, n_items))

# num_groups = int(np.floor(n_users / block_size))
#
# sorted_users = np.argsort(profile_length)
#
# # Now we plot the recommendation quality of recommenders
#
# MAP_itemKNNCF_per_group = []
# MAP_itemKNNCBF_per_group = []
# MAP_pureSVD_per_group = []
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
#     results, _ = evaluator_test.evaluateRecommender(itemKNNCF)
#     MAP_itemKNNCF_per_group.append(results[cutoff]["MAP"])
#
#     results, _ = evaluator_test.evaluateRecommender(pureSVD)
#     MAP_pureSVD_per_group.append(results[cutoff]["MAP"])
#
#     results, _ = evaluator_test.evaluateRecommender(itemKNNCBF)
#     MAP_itemKNNCBF_per_group.append(results[cutoff]["MAP"])
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
# pyplot.plot(MAP_itemKNNCF_per_group, label="itemKNNCF")
# pyplot.plot(MAP_itemKNNCBF_per_group, label="itemKNNCBF")
# pyplot.plot(MAP_pureSVD_per_group, label="pureSVD")
# pyplot.plot(MAP_topPop_per_group, label="topPop")
# pyplot.plot(MAP_itemKNN_scores_hybrid_per_group, label="ItemKNNCF + pureSVD")
# pyplot.ylabel('MAP')
# pyplot.xlabel('User Group')
# pyplot.legend()
# pyplot.show()


################################################################################################################

# LightFM's hybrid model

# Import the model
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, auc_score

# Set the number of threads; you can increase this
# ify you have more physical cores available.
# NUM_THREADS = 2
# NUM_COMPONENTS = 30
# NUM_EPOCHS = 3
# ITEM_ALPHA = 1e-6
#
# # Define a new model instance
# model = LightFM(loss='warp',
#                 item_alpha=ITEM_ALPHA,
#                 no_components=NUM_COMPONENTS)
#
# # Fit the hybrid model. Note that this time, we pass
# # in the item features matrix.
# model = model.fit(URM_train,
#                 # item_features=item_features,
#                 epochs=NUM_EPOCHS,
#                 num_threads=NUM_THREADS)
#
# # Evaluate the trained model
# train_precision = precision_at_k(model,
#                                 URM_train,
#                                 k=cutoff,
#                                 num_threads=NUM_THREADS).mean()
#
# test_precision = precision_at_k(model,
#                                 URM_test,
#                                 k=cutoff,
#                                 num_threads=NUM_THREADS).mean()
#
#
# print('Precision: train %.5f, test %.5f.' % (train_precision, test_precision))
# print('AUC: train %.5f, test %.5f.' % (train_auc, test_auc))


################################################################################################################


# RandomRecommender = RandomRecommender(URM_train)
# RandomRecommender.fit()
# cold_user_mask = RandomRecommender._get_cold_user_mask()
#
# # predictions
# user_recommendations_items = []
# user_recommendations_user_id = []
#
# for n_user in range(100):
#     recommendations = RandomRecommender.recommend(n_user,
#                                                   cutoff=cutoff)
#
#     user_recommendations_items.extend(recommendations)
#     user_recommendations_user_id.extend([n_user] * len(recommendations))
#
#
# # Add the prediction of another algorithm
# topPop = TopPopRecommender(URM_train)
# topPop.fit()
#
# topPop_score_list = []
#
# for user_id, item_id in zip(user_recommendations_user_id, user_recommendations_items):
#     topPop_score = topPop._compute_item_score([user_id])[0, item_id]
#     topPop_score_list.append(topPop_score)
#
#
# exit(0)
