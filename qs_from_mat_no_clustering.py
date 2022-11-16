import re
import os
import pickle
import swifter
import numpy as np
import pandas as pd
import sys
import joblib
import matplotlib.pyplot as plt
import datetime

from pandas import DataFrame
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn import metrics


PATH = "/vol/ek/Home/orlyl02/working_dir/oligopred/embed_transfer/"
cov_PATH = "/vol/ek/Home/orlyl02/working_dir/oligopred/embed_transfer/transfer_cov03/"
SAVE_PATH = cov_PATH
NUM_CV = 5


def get_data():
    overall_train_set = pd.read_pickle(cov_PATH + "train_set_c0.3.pkl")
    overall_train_set.reset_index(drop=True, inplace=True)
    train_set = pd.read_pickle(PATH + "train_set_c03_8020_train2.pkl")
    test_set = pd.read_pickle(PATH + "test_set_c03_8020_test2.pkl")
    return overall_train_set, train_set, test_set


def get_matrices():
    alnRes_mat = pd.read_pickle(PATH + "alnRes_blast1_mat.pkl")
    cosine_sim_df = pd.read_pickle(cov_PATH + "cosine_sim_df_c0.3.pkl")
    pdbs_not_in_blast = [x for x in cosine_sim_df.index.to_list() if x not in alnRes_mat.index.to_list()]
    pdbs_in_holdout = [x for x in alnRes_mat.index.to_list() if x not in overall_train_set.code.to_list()]
    pdbs_to_remove = pdbs_not_in_blast + pdbs_in_holdout
    alnRes_mat.drop(pdbs_to_remove, axis=0, errors='ignore', inplace=True)
    alnRes_mat.drop(pdbs_to_remove, axis=1, errors='ignore', inplace=True)
    cosine_sim_df.drop(pdbs_to_remove, axis=0, errors='ignore', inplace=True)
    cosine_sim_df.drop(pdbs_to_remove, axis=1, errors='ignore', inplace=True)
    return alnRes_mat, cosine_sim_df, pdbs_to_remove


def get_label_score_loop_seq(overall_train_set, ref_mat):
    for index, row in overall_train_set.iterrows():
        pdb_code = row["code"]
        ref_mat_copy = ref_mat.copy()
        ref_mat_copy.drop(pdb_code, axis=1, errors='ignore', inplace=True)
        ref = ref_mat_copy.loc[pdb_code].idxmax()
        label = float(overall_train_set[overall_train_set["code"] == ref].nsub)
        overall_train_set.loc[index, "alnRes_pred"] = label
        overall_train_set.loc[index, "alnRes_max_score"] = ref_mat_copy.loc[pdb_code].max()
    return overall_train_set


def get_label_score_loop_cos(overall_train_set, ref_mat):
    for index, row in overall_train_set.iterrows():
        pdb_code = row["code"]
        ref_mat_copy = ref_mat.copy()
        ref_mat_copy.drop(pdb_code, axis=1, errors='ignore', inplace=True)
        ref = ref_mat_copy.loc[pdb_code].idxmax()
        label = float(overall_train_set[overall_train_set["code"] == ref].nsub)
        overall_train_set.loc[index, "cos_pred"] = label
        overall_train_set.loc[index, "cos_max_score"] = ref_mat_copy.loc[pdb_code].max()
    return overall_train_set


def calc_model_params(final_pred):
    # pdbs_remove_from_test = [x for x in test_set.code.to_list() if x not in cosine_pred_df.code.to_list()]
    # test_set = test_set[~test_set.code.isin(pdbs_remove_from_test)]
    y_test = final_pred['nsub']
    cosine_f1_score = round(f1_score(y_test, final_pred.cos_pred, average="weighted"), 3)
    cosine_bal_acc = round(metrics.balanced_accuracy_score(y_test, final_pred.cos_pred, adjusted=True), 3)
    # cosine_majority_f1_score = round(f1_score(y_test, cosine_pred_df.cosine_majority_top10, average="weighted"), 3)
    # cosine_majority_bal_acc = round(metrics.balanced_accuracy_score(y_test, cosine_pred_df.cosine_majority_top10, adjusted=True), 3)
    seq_f1_score = round(f1_score(y_test, final_pred.alnRes_pred, average='weighted'), 3)
    seq_bal_acc = round(metrics.balanced_accuracy_score(y_test, final_pred.alnRes_pred, adjusted=True), 3)
    # seq_majority_f1_score = round(f1_score(y_test, alnRes_pred_df.alnRes_majority_top10, average='weighted'), 3)
    # seq_majority_bal_acc = round(metrics.balanced_accuracy_score(y_test, alnRes_pred_df.alnRes_majority_top10, adjusted=True), 3)

    with open(SAVE_PATH + "score_results_no_clust.csv", 'w') as f:
        f.write("Cosine_F1_score: " + str(cosine_f1_score) + "\n")
        f.write("Cosine_balanced_accuracy: " + str(cosine_bal_acc) + "\n")
        # f.write("Cosine_majority_F1_score: " + str(cosine_majority_f1_score) + "\n")
        # f.write("Cosine_majority_balanced_accuracy: " + str(cosine_majority_bal_acc) + "\n")
        f.write("Seq_F1_score: " + str(seq_f1_score) + "\n")
        f.write("Seq_balanced_accuracy: " + str(seq_bal_acc) + "\n")
        # f.write("Seq_majority_F1_score: " + str(seq_majority_f1_score) + "\n")
        # f.write("Seq_majority_balanced_accuracy: " + str(seq_majority_bal_acc) + "\n")

    return y_test



# def get_qs_from_seq(overall_train_set, alnRes_mat):
#     pred = overall_train_set.copy()
#     alnRes_mat = alnRes_mat.sparse.to_dense()
#     pred["alnRes_pred"] = pred.apply(lambda row: get_label(row, alnRes_mat), axis=1)
#     pred["alnRes_max_score"] = pred.apply(lambda row: get_max_score(row, alnRes_mat), axis=1)
#     print("finished alnRes")
#     return pred
#
# def get_qs_from_cos(overall_train_set, cosine_sim_df):
#     pred = overall_train_set.copy()
#     pred["cos_pred"] = pred.apply(lambda row: get_label(row, cosine_sim_df), axis=1)
#     pred["cos_max_score"] = pred.apply(lambda row: get_max_score(row, cosine_sim_df), axis=1)
#     print("finished cosine")
#     return pred
#

# def get_label(row, ref_mat):
#     pdb_code = row["code"]
#     print("pdb_code", pdb_code)
#     ref_mat.drop(pdb_code, axis=1, errors='ignore', inplace=True)
#     print(ref_mat.shape)
#     ref = ref_mat.loc[pdb_code].idxmax()
#     print("ref", ref, flush=True)
#     # this is the first entry, thus if the max is 0, it means there is no match
#     if ref == "5ahz_1":
#         if ref_mat.loc[pdb_code].max() == 0.000:
#             label = 0.0
#     else:
#         # print((overall_train_set[overall_train_set["code"] == ref].nsub))
#         label = float(overall_train_set[overall_train_set["code"] == ref].nsub)
#     # print(label, flush=True)
#     # print(ref_mat.loc[pdb_code].max())
#     return label
#
#
# def get_majority(row, ref_mat):
#     pdb_code = row["code"]
#     ref_list = ref_mat.loc[pdb_code].nlargest(10).index.tolist()
#     label_list = []
#     for pdb_ref in ref_list:
#         label = float(train_set[train_set["code"] == pdb_ref].nsub)
#         label_list.append(label)
#     top10_label = max(set(label_list), key=label_list.count)
#     return top10_label
#
#
# def get_max_score(row, ref_mat):
#     pdb_code = row["code"]
#     ref_mat.drop(pdb_code, axis=1, errors='ignore', inplace=True)
#     return ref_mat.loc[pdb_code].max()


if __name__ == "__main__":
    overall_train_set, train_set, test_set = get_data()
    alnRes_mat, cosine_sim_df, pdbs_to_remove = get_matrices()
    filtered_overall_train_set = overall_train_set[~overall_train_set["code"].isin(pdbs_to_remove)]
    filtered_overall_train_set.reset_index(drop=True, inplace=True)
    overall_with_seq_pred = get_label_score_loop_seq(filtered_overall_train_set, alnRes_mat)
    final_pred = get_label_score_loop_cos(overall_with_seq_pred, cosine_sim_df)
    y_test = calc_model_params(final_pred)
    final_pred.to_pickle(SAVE_PATH + "final_pred_no_clust.pkl")
    # final_pred = pd.read_pickle(SAVE_PATH + "final_pred_no_clust.pkl")

    print(final_pred)

