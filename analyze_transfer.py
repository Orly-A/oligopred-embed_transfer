import re
import os
import pickle
import numpy as np
import pandas as pd

import seaborn as sns
from sklearn import metrics

PATH = "/vol/ek/Home/orlyl02/working_dir/oligopred/embed_transfer/"
cov_PATH = "/vol/ek/Home/orlyl02/working_dir/oligopred/embed_transfer/transfer_cov03/"
SAVE_PATH = cov_PATH


def get_le_dict():
    with open("/vol/ek/Home/orlyl02/working_dir/oligopred/xgboost/le_dict.pkl", 'rb') as f:
        le_dict = pickle.load(f)
    inv_map = {v: k for k, v in le_dict.items()}
    return inv_map


def gen_conf_mat(transfer_preds, inv_map_seq, inv_map_cos):
    conf_mat_cosine = metrics.confusion_matrix(transfer_preds["nsub"], transfer_preds["cos_pred"])
    analyze_save_conf_mat(conf_mat_cosine, inv_map_cos, "cosine_pred_conf_mat_no_clust")
    conf_mat_seq = metrics.confusion_matrix(transfer_preds["nsub"], transfer_preds["alnRes_pred"])
    analyze_save_conf_mat(conf_mat_seq, inv_map_seq, "alnRes_pred_conf_mat_no_clust")



def analyze_save_conf_mat(conf_mat, inv_map, name):
    conf_mat_df = pd.DataFrame(conf_mat)
    conf_mat_df.rename(inv_map, axis=1, inplace=True)
    conf_mat_df.rename(inv_map, inplace=True)
    conf_mat_df_percent = conf_mat_df.div(conf_mat_df.sum(axis=1), axis=0).round(2)
    s = sns.heatmap(conf_mat_df_percent, annot=True, fmt='g', cmap="BuPu", annot_kws={"size": 6})
    s.set(xlabel='Prediction', ylabel='Actual lables')
    s.figure.savefig(SAVE_PATH + name + ".png")
    s.figure.clf()


if __name__ == "__main__":
    # alnRes_pred_df = pd.read_pickle(PATH + "alnRes_pred_df.pkl")
    # cosine_pred_df = pd.read_pickle(PATH + "cosine_pred_df.pkl")
    alnRes_pred_df = pd.read_pickle(cov_PATH + "alnRes_blast1_pred_df_c03.pkl")
    cosine_pred_df = pd.read_pickle(cov_PATH + "cosine_pred_df_c03.pkl")
    transfer_preds = pd.read_pickle(cov_PATH + "final_pred_no_clust.pkl")
    # transfer_preds = pd.merge(alnRes_pred_df, cosine_pred_df, how="outer", on=("code", "nsub"))
    transfer_preds_no_zeros = transfer_preds[transfer_preds["alnRes_pred"] != 0.0]
    inv_map = get_le_dict()
    # inv_map_pred_seq = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 10, 10: 12, 11: 14, 12: 24}
    # inv_map_no_zeros_seq = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 14, 13: 15, 14: 16, 15: 24, 16: 60}
    inv_map_no_zeros_transfer = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 18, 17: 24, 18: 60}
    # transfer_preds_remove_small = transfer_preds[~transfer_preds["nsub"].isin([13.0, 18.0])]
    # gen_conf_mat(transfer_preds_remove_small, inv_map_no_zeros_seq, inv_map_no_zeros_seq)
    gen_conf_mat(transfer_preds, inv_map_no_zeros_transfer, inv_map_no_zeros_transfer)
    print("done")


