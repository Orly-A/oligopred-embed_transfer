import re
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from scipy.stats import ranksums


PATH = "/vol/ek/Home/orlyl02/working_dir/oligopred/embed_transfer/"
# cov_PATH = "/vol/ek/Home/orlyl02/working_dir/oligopred/embed_transfer/transfer_cov03/"
# for use with esm embeddings
cov_PATH = "/vol/ek/Home/orlyl02/working_dir/oligopred/embed_transfer/esm_embeds/"

SAVE_PATH = cov_PATH


def get_le_dict():
    with open("/vol/ek/Home/orlyl02/working_dir/oligopred/xgboost/le_dict.pkl", 'rb') as f:
        le_dict = pickle.load(f)
    inv_map = {v: k for k, v in le_dict.items()}
    return inv_map


def gen_conf_mat(transfer_preds, inv_map_seq, inv_map_cos):
    transfer_preds.dropna(inplace=True)
    conf_mat_cosine = metrics.confusion_matrix(transfer_preds["nsub"], transfer_preds["cosine_pred"])
    analyze_save_conf_mat(conf_mat_cosine, inv_map_cos, "cosine_pred_conf_mat_with_clust_esm")
    print("cosine balanced accuracy, not adjusted:")
    print(round(metrics.balanced_accuracy_score(transfer_preds["nsub"], transfer_preds["cosine_pred"]), 3))
    conf_mat_seq = metrics.confusion_matrix(transfer_preds["nsub"], transfer_preds["alnRes_pred"])
    analyze_save_conf_mat(conf_mat_seq, inv_map_seq, "alnRes_pred_conf_mat_with_clust_esm")
    print("sequence balanced accuracy, not adjusted:")
    print(round(metrics.balanced_accuracy_score(transfer_preds["nsub"], transfer_preds["alnRes_pred"]), 3))


def analyze_save_conf_mat(conf_mat, inv_map, name):
    conf_mat_df = pd.DataFrame(conf_mat)
    conf_mat_df.rename(inv_map, axis=1, inplace=True)
    conf_mat_df.rename(inv_map, inplace=True)
    conf_mat_df_percent = conf_mat_df.div(conf_mat_df.sum(axis=1), axis=0).round(2)
    s = sns.heatmap(conf_mat_df_percent, annot=True, fmt='g', cmap="BuPu", annot_kws={"size": 8}, vmin=0, vmax=1)
    s.set(xlabel='Prediction', ylabel='Actual lables', title=name)
    s.figure.savefig(SAVE_PATH + name + ".png")
    s.figure.clf()


def proba_dist_right_wrong(transfer_preds):
    distribution_df = pd.DataFrame(index=range(transfer_preds.shape[0]))
    for chosen_nsub in sorted(transfer_preds.nsub.unique().tolist()):
        chosen_df = transfer_preds[transfer_preds["cos_pred"] == chosen_nsub][["nsub", "cos_pred", "cos_max_score"]]
        chosen_df_true = chosen_df[chosen_df["nsub"] == chosen_nsub]
        chosen_df_wrong = chosen_df[chosen_df["nsub"] != chosen_nsub]
        distribution_df[str(int(chosen_nsub)) + "_correct"] = chosen_df_true["cos_max_score"]
        distribution_df[str(int(chosen_nsub)) + "_wrong"] = chosen_df_wrong["cos_max_score"]
        print(ranksums(chosen_df_true["cos_max_score"], chosen_df_wrong["cos_max_score"], alternative='greater'),
              "wilcoxon for " + str(int(chosen_nsub)))
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.setp(sns.boxplot(data=distribution_df, palette=["darkturquoise", "darkorchid"]).get_xticklabels(), rotation=45)
    ax.set_ylabel("probability")
    ax.set_title("'Confidence' in predicted label when correct (green) or wrong (purple)")
    plt.ylim(0.84, 1.01)
    plt.yticks([0.85, 0.9, 0.95, 1.0])
    # plt.show()
    plt.savefig(cov_PATH + "probability_each_label_for_true_and_false_positives_capped0.85.png")
    plt.close()



if __name__ == "__main__":
    alnRes_pred_df = pd.read_pickle("/vol/ek/Home/orlyl02/working_dir/oligopred/embed_transfer/transfer_cov03/alnRes_blast1_pred_df_c03.pkl")
    cosine_pred_df = pd.read_pickle(cov_PATH + "cosine_pred_df_c03_esm.pkl")
    # alnRes_pred_df = pd.read_pickle(cov_PATH + "alnRes_blast1_pred_df_c03.pkl")
    # cosine_pred_df = pd.read_pickle(cov_PATH + "cosine_pred_df_c03_esm.pkl")
    transfer_preds = pd.read_pickle(cov_PATH + "final_pred_no_clust_esm.pkl")
    # transfer_preds = pd.merge(alnRes_pred_df, cosine_pred_df, how="outer", on=("code", "nsub"))
    transfer_preds_no_zeros = transfer_preds[transfer_preds["alnRes_pred"] != 0.0]
    inv_map = get_le_dict()
    # inv_map_pred_seq = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 10, 10: 12, 11: 14, 12: 24}
    # inv_map_no_zeros_seq = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 14, 13: 15, 14: 16, 15: 24, 16: 60}
    # inv_map_no_zeros_transfer = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 18, 17: 24, 18: 60}
    inv_map_cos = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 8, 7: 10, 8: 12, 9: 24}
    inv_map_cos = inv_map
    inv_map_seq = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 10, 10: 12, 11: 14, 12: 24}
    # transfer_preds_remove_small = transfer_preds[~transfer_preds["nsub"].isin([13.0, 18.0])]
    # gen_conf_mat(transfer_preds_remove_small, inv_map_no_zeros_seq, inv_map_no_zeros_seq)
    # gen_conf_mat(transfer_preds, inv_map_seq, inv_map_cos)
    print("done")


