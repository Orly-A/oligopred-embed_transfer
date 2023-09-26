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

PATH = "/vol/ek/Home/orlyl02/working_dir/oligopred/embed_transfer/"
# for use with esm embeddings
cov_PATH = "/vol/ek/Home/orlyl02/working_dir/oligopred/embed_transfer/esm_embeds/"
SAVE_PATH = "/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/holdout_analysis/"
NUM_CV = 5



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


def get_qs_from_embed(cosine_sim_df, holdout_set, train_set):
    pred = holdout_set.copy()
    pred["cosine_pred"] = pred.apply(lambda row: get_label(row, cosine_sim_df[train_set["code"].to_list()], train_set), axis=1)
    pred["cosine_max_score"] = pred.apply(lambda row: get_max_score(row, cosine_sim_df[train_set["code"].to_list()]), axis=1)
    # pred["cosine_majority_top10"] = pred.apply(lambda row: get_majority(row, cosine_sim_df[train_set["code"].to_list()]), axis=1)
    print("finished cosine")
    return pred


def get_qs_from_seq(alnRes_mat, holdout_set, train_set):
    pred = holdout_set.copy()
    # # pred.drop(pdbs_not_in_train, axis=0, inplace=True, errors='ignore') # row below fixes this
    # pred = pred[~pred.code.isin(pdbs_not_in_blast)]
    # # alnRes_mat_remove_test_from_cols = alnRes_mat.drop(X_test, axis=1)
    # alnRes_mat_remove_test_from_cols = alnRes_mat.drop(pdbs_not_in_train, axis=1, errors='ignore')
    pred["alnRes_pred"] = pred.apply(lambda row: get_label(row, alnRes_mat[alnRes_mat.columns.intersection(train_set["code"].to_list())], train_set), axis=1)
    pred["alnRes_max_score"] = pred.apply(lambda row: get_max_score(row, alnRes_mat[alnRes_mat.columns.intersection(train_set["code"].to_list())]), axis=1)
    # alnRes_mat_remove_test_from_cols_dense = alnRes_mat_remove_test_from_cols.sparse.to_dense()
    # pred["alnRes_majority_top10"] = pred.apply(lambda row: get_majority(row, alnRes_mat_remove_test_from_cols_dense), axis=1)
    print("finished alnRes")
    return pred


def get_label(row, ref_mat, train_set):
    pdb_code = row["code"]
    if pdb_code == "1zzn_1":
        return np.nan
    ref = ref_mat.loc[pdb_code].idxmax()
    label = float(train_set[train_set["code"] == ref].nsub)
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
    cosine_bal_acc = round(metrics.balanced_accuracy_score(y_test, cosine_pred_df.cosine_pred), 3)
    cosine_majority_f1_score = round(f1_score(y_test, cosine_pred_df.cosine_majority_top10, average="weighted"), 3)
    cosine_majority_bal_acc = round(metrics.balanced_accuracy_score(y_test, cosine_pred_df.cosine_majority_top10), 3)
    seq_f1_score = round(f1_score(y_test, alnRes_pred_df.alnRes_pred, average='weighted'), 3)
    seq_bal_acc = round(metrics.balanced_accuracy_score(y_test, alnRes_pred_df.alnRes_pred), 3)
    seq_majority_f1_score = round(f1_score(y_test, alnRes_pred_df.alnRes_majority_top10, average='weighted'), 3)
    seq_majority_bal_acc = round(metrics.balanced_accuracy_score(y_test, alnRes_pred_df.alnRes_majority_top10), 3)

    with open(SAVE_PATH + "score_results_esm.csv", 'w') as f:
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
    cosine_sim_df = pd.read_pickle(cov_PATH + "cosine_sim_df_holdout_c0.3_esm.pkl")
    # cosine_sim_df = pd.read_pickle(cov_PATH + "cosine_sim_df_c0.3_esm.pkl")
    # train_set, X_train, y_train, test_set, X_test, y_test, list_of_pdbs_to_remove = data_definition()

    # pdbs_not_in_train = [x for x in alnRes_mat.columns.to_list() if x not in train_set.code.to_list()]
    # to be honest I am not sure why excatly we are missing these, I suspect they dont have any blast hits but haven't completely verified
    # pdbs_not_in_blast = [x for x in cosine_sim_df.index.to_list() if x not in alnRes_mat.index.to_list()]
    # list_for_cosine = list(set(list_of_pdbs_to_remove + pdbs_not_in_blast))
    # list_for_seq = list(set(pdbs_not_in_train + pdbs_not_in_blast))
    holdout_set = pd.read_pickle("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/holdout_analysis/overall_proba_pred_ecod_holdout.pkl")
    train_set = pd.read_csv("/vol/ek/Home/orlyl02/working_dir/oligopred/mlp/esm_runs/5cv_sets_proba_models/overall_proba_pred_ecod.csv", sep='\t')
    # holdout_set = get_qs_from_embed(cosine_sim_df, holdout_set, train_set)
    holdout_set = get_qs_from_seq(alnRes_mat, holdout_set, train_set)
    # cosine_pred_df.to_pickle(PATH + "cosine_pred_df_2.pkl")
    # alnRes_pred_df.to_pickle(PATH + "alnRes_pred_df_2.pkl")

    holdout_set.to_pickle(SAVE_PATH + "holdout_set_embed_transfer.pkl")
    # alnRes_pred_df.to_pickle(SAVE_PATH + "alnRes_blast_e100_pred_df_c03.pkl")
    # dont need to run this again, loaded the saved files

    alnRes_pred_df = pd.read_pickle("/vol/ek/Home/orlyl02/working_dir/oligopred/embed_transfer/transfer_cov03/alnRes_blast1_pred_df_c03.pkl")
    lst_not_in_cos = [x for x in y_test.index.to_list() if x not in cosine_pred_df.index.to_list()]
    lst_not_in_aln = [x for x in y_test.index.to_list() if x not in alnRes_pred_df.index.to_list()]
    y_test.drop(lst_not_in_cos, inplace=True, errors='ignore')
    y_test.drop(lst_not_in_aln, inplace=True, errors='ignore')
    cosine_pred_df.drop(lst_not_in_aln, inplace=True, errors='ignore')
    alnRes_pred_df.drop(lst_not_in_cos, inplace=True, errors='ignore')
    y_test = calc_model_params(cosine_pred_df, alnRes_pred_df, test_set)
    print("Cosine_F1_score: " + str(round(f1_score(y_test, cosine_pred_df.cosine_pred, average="weighted"), 3)))
    print("Seq_F1_score: " + str(round(f1_score(y_test, alnRes_pred_df.alnRes_pred, average='weighted'), 3)))
