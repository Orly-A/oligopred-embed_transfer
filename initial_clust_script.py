import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold

PATH = "/vol/ek/Home/orlyl02/working_dir/oligopred/"


def read_cluster_tab():
    cluster_tab = pd.read_csv("/vol/ek/Home/orlyl02/working_dir/oligopred/clustering/re_clust_c0.3/session_cluster.tsv", sep='\t', header=None)
    cluster_tab.rename(columns={0: "representative", 1: "code"}, inplace=True)
    cluster_tab.drop_duplicates(subset=(["code"]), inplace=True)
    return cluster_tab


def count_clusters(full_tab_with_clusters):
    # interesting comparison of the number of clusters per nsub, resulting from the different clustering
    list_of_nsubs = list(set(full_tab_with_clusters["nsub"].tolist()))
    print("nsub", "num_of_clusts", "num_of_codes")
    for nsub in list_of_nsubs:
        num_of_clusts = full_tab_with_clusters[full_tab_with_clusters['nsub'] == nsub].groupby("representative").nunique().shape[0]
        print(nsub, num_of_clusts, full_tab_with_clusters.groupby("nsub").size()[nsub])


def get_x_y(full_tab_with_clusters):
    X = full_tab_with_clusters["code"]
    y = full_tab_with_clusters["nsub"]
    groups = full_tab_with_clusters["representative"]
    return X, y, groups

def split_data(X, y, groups, full_tab_with_clusters):
    cv = StratifiedGroupKFold(n_splits=10)
    train_lst = []
    test_lst = []
    for train_idxs, test_idxs in cv.split(X, y, groups):
        train_lst.append(X[train_idxs].tolist())
        test_lst.append(X[test_idxs].tolist())

    train_idx_df = pd.DataFrame(train_lst).transpose()
    train_idx_df.rename(columns={0:"train_0", 1:"train_1", 2:"train_2", 3:"train_3", 4:"train_4", 5:"train_5", 6:"train_6", 7:"train_7", 8:"train_8", 9:"train_9"}, inplace=True)
    # print(train_idx_df)
    test_idx_df = pd.DataFrame(test_lst).transpose()
    test_idx_df.rename(columns={0:"test_0", 1:"test_1", 2:"test_2", 3:"test_3", 4:"test_4", 5:"test_5", 6:"test_6", 7:"test_7", 8:"test_8", 9:"test_9"}, inplace=True)
    # print(test_idx_df)

    merged_train_test = pd.concat([train_idx_df, test_idx_df], axis=1, join="outer")
    train_set = full_tab_with_clusters[full_tab_with_clusters["code"].isin(merged_train_test["train_0"])]
    hold_out_set = full_tab_with_clusters[full_tab_with_clusters["code"].isin(merged_train_test["test_0"])]
    # save the train and hold out sets to pickle files
    train_set.to_pickle(PATH + "clustering/re_clust_c0.3/train_set_c0.3.pkl")
    hold_out_set.to_pickle(PATH + "clustering/re_clust_c0.3/hold_out_set_c0.3.pkl")
    return train_set, hold_out_set


if __name__ == "__main__":
    full_tab_for_embed = pd.read_pickle(
        "/vol/ek/Home/orlyl02/working_dir/oligopred/clustering/parsed_tab_for_embed.pkl")
    cluster_tab = read_cluster_tab()
    full_tab_with_clusters = pd.merge(full_tab_for_embed, cluster_tab, on="code", how="outer")
    X, y, groups = get_x_y(full_tab_with_clusters)
    with open(PATH + "clustering/embed_pkl_final_all", "rb") as f:
        npy_embed_all = pickle.load(f)
    filt_npy_embed_all = npy_embed_all[1:]
    full_tab_with_clusters['embeddings'] = list(filt_npy_embed_all.astype("float32"))
    full_tab_with_clusters.to_pickle(PATH + "clustering/re_clust_c0.3/full_tab_with_clust_embed.pkl")
    train_set, hold_out_set = split_data(X, y, groups, full_tab_with_clusters)
    print(train_set, hold_out_set)

