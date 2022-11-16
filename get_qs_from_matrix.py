import re
import os
import pickle
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


from io import StringIO

PATH = "/vol/ek/Home/orlyl02/working_dir/oligopred/embed_transfer/"
cov_PATH = "/vol/ek/Home/orlyl02/working_dir/oligopred/embed_transfer/transfer_cov03/"
SAVE_PATH = cov_PATH
NUM_CV = 5


def data_definition():
    # with open(PATH + "train_set.pkl", 'rb') as f:
    with open(cov_PATH + "train_set_c0.3.pkl", 'rb') as f:
        overall_train_set = pickle.load(f)
    # index reset is important for the stratified splitting and the saving to lists
    overall_train_set.reset_index(drop=True, inplace=True)

    overall_train_set, list_of_pdbs_to_remove = remove_small_groups(overall_train_set)
    overall_train_set.reset_index(drop=True, inplace=True)
    X = overall_train_set["code"]
    y = overall_train_set["nsub"]
    groups = overall_train_set["representative"]

    cv = StratifiedGroupKFold(n_splits=NUM_CV, shuffle=True, random_state=1)
    train_lst = []
    test_lst = []
    for train_idxs, test_idxs in cv.split(X, y, groups):
        train_lst.append(X[train_idxs].tolist())
        test_lst.append(X[test_idxs].tolist())

    train_idx_df = pd.DataFrame(train_lst).transpose()
    train_idx_df.rename(columns={0:"train_0", 1:"train_1", 2:"train_2", 3:"train_3", 4:"train_4"}, inplace=True)
    test_idx_df = pd.DataFrame(test_lst).transpose()
    test_idx_df.rename(columns={0:"test_0", 1:"test_1", 2:"test_2", 3:"test_3", 4:"test_4"}, inplace=True)

    merged_train_test = pd.concat([train_idx_df, test_idx_df], axis=1, join="outer")
    # using set number 2 since the sizes of the folds are very different and test_0 has only 38 pdbs...
    train_set = overall_train_set[overall_train_set["code"].isin(merged_train_test["train_2"])]
    test_set = overall_train_set[overall_train_set["code"].isin(merged_train_test["test_2"])]
    train_set.to_pickle(PATH + "train_set_c03_8020_train2.pkl")
    test_set.to_pickle(PATH + "test_set_c03_8020_test2.pkl")

    # Differently from the original code, I'm using 'code' and not 'embeddings' as X
    train_set = train_set[["code", "nsub"]]
    test_set = test_set[["code", "nsub"]]
    X_train = train_set['code'].tolist()
    y_train = train_set['nsub']
    X_test = test_set['code'].tolist()
    y_test = test_set['nsub']

    return train_set, X_train, y_train, test_set, X_test, y_test, list_of_pdbs_to_remove


def remove_small_groups(overall_train_set):
    overall_train_set_no_embed = overall_train_set[["code", "nsub", "representative"]]
    overall_train_set2 = overall_train_set.copy()
    list_of_nsubs = list(set(overall_train_set2["nsub"].tolist()))
    list_of_pdbs_to_remove = []
    for nsub in list_of_nsubs:
        num_of_clusts = overall_train_set_no_embed[overall_train_set_no_embed['nsub'] == nsub].groupby("representative").nunique().shape[0]
        if num_of_clusts < NUM_CV:
            print(nsub, "nsub")
            print(num_of_clusts, "num_of_clusts")
            overall_train_set2 = overall_train_set2[overall_train_set2.nsub != nsub]
            list_of_pdbs_to_remove.extend(overall_train_set_no_embed[overall_train_set_no_embed['nsub'] == nsub].code.to_list())
    return overall_train_set2, list_of_pdbs_to_remove


def get_qs_from_embed(test_set, cosine_sim_df, list_of_pdbs_to_remove,pdbs_not_in_blast):
    pred = test_set.copy()
    # pred.drop(list_of_pdbs_to_remove, axis=0, inplace=True, errors='ignore')
    pred = pred[~pred.code.isin(pdbs_not_in_blast)]

    cosine_sim_df_remove_test_from_cols = cosine_sim_df.drop(X_test, axis=1)
    cosine_sim_df_remove_test_from_cols.drop(list_of_pdbs_to_remove, axis=1, inplace=True, errors='ignore')
    pred["cosine_pred"] = pred.apply(lambda row: get_label(row, cosine_sim_df_remove_test_from_cols), axis=1)
    pred["cosine_max_score"] = pred.apply(lambda row: get_max_score(row, cosine_sim_df_remove_test_from_cols), axis=1)
    pred["cosine_majority_top10"] = pred.apply(lambda row: get_majority(row, cosine_sim_df_remove_test_from_cols), axis=1)
    print("finished cosine")

    return pred


def get_qs_from_seq(test_set, alnRes_mat, pdbs_not_in_train, pdbs_not_in_blast):
    pred = test_set.copy()
    # pred.drop(pdbs_not_in_train, axis=0, inplace=True, errors='ignore') # row below fixes this
    pred = pred[~pred.code.isin(pdbs_not_in_blast)]
    # alnRes_mat_remove_test_from_cols = alnRes_mat.drop(X_test, axis=1)
    alnRes_mat_remove_test_from_cols = alnRes_mat.drop(pdbs_not_in_train, axis=1, errors='ignore')
    pred["alnRes_pred"] = pred.apply(lambda row: get_label(row, alnRes_mat_remove_test_from_cols), axis=1)
    pred["alnRes_max_score"] = pred.apply(lambda row: get_max_score(row, alnRes_mat_remove_test_from_cols), axis=1)
    alnRes_mat_remove_test_from_cols_dense = alnRes_mat_remove_test_from_cols.sparse.to_dense()
    pred["alnRes_majority_top10"] = pred.apply(lambda row: get_majority(row, alnRes_mat_remove_test_from_cols_dense), axis=1)
    print("finished alnRes")
    return pred


def get_label(row, ref_mat):
    pdb_code = row["code"]
    ref = ref_mat.loc[pdb_code].idxmax()
    # this is the first enty, thus if the max is 0, it means there is no match
    if ref == "5ahz_1":
        if ref_mat.loc[pdb_code].max() == 0.000:
            label = 0.0
    else:
        label = float(train_set[train_set["code"] == ref].nsub)
    # print(label)
    return label


def get_majority(row, ref_mat):
    pdb_code = row["code"]
    ref_list = ref_mat.loc[pdb_code].nlargest(10).index.tolist()
    label_list = []
    for pdb_ref in ref_list:
        label = float(train_set[train_set["code"] == pdb_ref].nsub)
        label_list.append(label)
    top10_label = max(set(label_list), key=label_list.count)
    return top10_label


def get_max_score(row, ref_mat):
    pdb_code = row["code"]
    return ref_mat.loc[pdb_code].max()



def calc_model_params(cosine_pred_df, alnRes_pred_df, test_set):
    pdbs_remove_from_test = [x for x in test_set.code.to_list() if x not in cosine_pred_df.code.to_list()]
    test_set = test_set[~test_set.code.isin(pdbs_remove_from_test)]
    y_test = test_set['nsub']
    cosine_f1_score = round(f1_score(y_test, cosine_pred_df.cosine_pred, average="weighted"), 3)
    cosine_bal_acc = round(metrics.balanced_accuracy_score(y_test, cosine_pred_df.cosine_pred, adjusted=True), 3)
    cosine_majority_f1_score = round(f1_score(y_test, cosine_pred_df.cosine_majority_top10, average="weighted"), 3)
    cosine_majority_bal_acc = round(metrics.balanced_accuracy_score(y_test, cosine_pred_df.cosine_majority_top10, adjusted=True), 3)
    seq_f1_score = round(f1_score(y_test, alnRes_pred_df.alnRes_pred, average='weighted'), 3)
    seq_bal_acc = round(metrics.balanced_accuracy_score(y_test, alnRes_pred_df.alnRes_pred, adjusted=True), 3)
    seq_majority_f1_score = round(f1_score(y_test, alnRes_pred_df.alnRes_majority_top10, average='weighted'), 3)
    seq_majority_bal_acc = round(metrics.balanced_accuracy_score(y_test, alnRes_pred_df.alnRes_majority_top10, adjusted=True), 3)

    with open(SAVE_PATH + "score_results_blast1.csv", 'w') as f:
        f.write("Cosine_F1_score: " + str(cosine_f1_score) + "\n")
        f.write("Cosine_balanced_accuracy: " + str(cosine_bal_acc) + "\n")
        f.write("Cosine_majority_F1_score: " + str(cosine_majority_f1_score) + "\n")
        f.write("Cosine_majority_balanced_accuracy: " + str(cosine_majority_bal_acc) + "\n")
        f.write("Seq_F1_score: " + str(seq_f1_score) + "\n")
        f.write("Seq_balanced_accuracy: " + str(seq_bal_acc) + "\n")
        f.write("Seq_majority_F1_score: " + str(seq_majority_f1_score) + "\n")
        f.write("Seq_majority_balanced_accuracy: " + str(seq_majority_bal_acc) + "\n")

    return y_test




if __name__ == "__main__":
    alnRes_mat = pd.read_pickle(PATH + "alnRes_blast_e100_mat.pkl")
    # cosine_sim_df = pd.read_pickle(PATH + "cosine_sim_df.pkl")
    cosine_sim_df = pd.read_pickle(cov_PATH + "cosine_sim_df_c0.3.pkl")
    train_set, X_train, y_train, test_set, X_test, y_test, list_of_pdbs_to_remove = data_definition()
    pdbs_not_in_train = [x for x in alnRes_mat.columns.to_list() if x not in train_set.code.to_list()]
    # to be honest I am not sure why excatly we are missing these, I suspect they dont have any blast hits but haven't completely verified
    pdbs_not_in_blast = [x for x in cosine_sim_df.index.to_list() if x not in alnRes_mat.index.to_list()]
    list_for_cosine = list(set(list_of_pdbs_to_remove + pdbs_not_in_blast))
    list_for_seq = list(set(pdbs_not_in_train + pdbs_not_in_blast))
    cosine_pred_df = get_qs_from_embed(test_set, cosine_sim_df, list_for_cosine, pdbs_not_in_blast)
    alnRes_pred_df = get_qs_from_seq(test_set, alnRes_mat, list_for_seq, pdbs_not_in_blast)
    # cosine_pred_df.to_pickle(PATH + "cosine_pred_df_2.pkl")
    # alnRes_pred_df.to_pickle(PATH + "alnRes_pred_df_2.pkl")

    cosine_pred_df.to_pickle(SAVE_PATH + "cosine_pred_df_c03.pkl")
    alnRes_pred_df.to_pickle(SAVE_PATH + "alnRes_blast_e100__pred_df_c03.pkl")

    y_test = calc_model_params(cosine_pred_df, alnRes_pred_df, test_set)
    print("Cosine_F1_score: " + str(round(f1_score(y_test, cosine_pred_df.cosine_pred, average="weighted"), 3)))
    print("Seq_F1_score: " + str(round(f1_score(y_test, alnRes_pred_df.alnRes_pred, average='weighted'), 3)))
