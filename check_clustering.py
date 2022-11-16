import pickle
import numpy as np
import pandas as pd
import swifter


PATH = "/vol/ek/Home/orlyl02/working_dir/oligopred/embed_transfer/"
PATH_TO_TRAIN_SET = "/vol/ek/Home/orlyl02/working_dir/oligopred/clustering/re_clust_c0.3/full_tab_with_clust_embed.pkl"


def get_files():
    alnRes_mat = pd.read_pickle(PATH + "alnRes_second_mat.pkl")
#    try:
#        alnRes_mat = pd.read_pickle(PATH + "alnRes_second_mat_dense.pkl")
#   except:
#        alnRes_mat = alnRes_mat.sparse.to_dense()
#        alnRes_mat.to_pickle(PATH + "alnRes_second_mat_dense.pkl")
    alnRes_mat = alnRes_mat.sparse.to_dense()
    alnRes_pred_df = pd.read_pickle(PATH + "alnRes_pred_df_2.pkl")
    cosine_sim_df = pd.read_pickle(PATH + "cosine_sim_df.pkl")
    cosine_pred_df = pd.read_pickle(PATH + "cosine_pred_df_2.pkl")
    with open(PATH_TO_TRAIN_SET, 'rb') as f:
        overall_train_set = pickle.load(f)
    # index reset is important for the stratified splitting and the saving to lists
    overall_train_set.reset_index(drop=True, inplace=True)
    overall_set = overall_train_set[["code", "nsub", "representative"]]
    return overall_train_set, overall_set, alnRes_mat, alnRes_pred_df, cosine_sim_df, cosine_pred_df


# def get_closest(overall_set, alnRes_mat):
#     overall_set.loc[:, "closest"] = overall_set.apply(lambda row: get_label(row, alnRes_mat), axis=1)
#     # overall_set.loc[:, "closest"] = overall_set.swifter.apply(lambda row: get_label(row, alnRes_mat), axis=1)
#     return overall_set


def get_label(row, ref_mat):
    pdb_code = row["code"]
    # print(pdb_code)
    # print(row["representative"])
    ref = ref_mat.loc[pdb_code].idxmax()
    if ref_mat.loc[pdb_code].idxmax() == pdb_code:
        ref = ref_mat.loc[pdb_code].nlargest(2).index[1]
    # print(ref)
    return ref





def get_highest_from_dif_clust(row, overall_set, alnRes_mat):
    pdb_code = row["code"]
    current_rep = overall_set[overall_set.code == pdb_code]["representative"].values[0]
    current_list = overall_set[overall_set.representative == current_rep].code.to_list()
    current_alnRes_mat = alnRes_mat.drop(current_list, axis=1)
    return current_alnRes_mat.loc[pdb_code].max()






def get_clust_label(overall_set):
    # Does not work
    overall_set.loc[:, "clust_label"] = overall_set.apply(lambda row: get_rep_label(row, overall_set), axis=1)
    # overall_set.loc[:, "clust_label"] = overall_set.swifter.apply(lambda row: get_rep_label(row, overall_set), axis=1)

    return overall_set



def get_rep_label(row, overall_set):
    relevant_code = row["closest"]
    print(relevant_code, "relevant_code")
    # if overall_set[overall_set.code == relevant_code]["representative"].values[0]:
    rep = overall_set[overall_set.code == relevant_code]["representative"].values[0]
 #   else:
#        rep = np.Nan
    print(rep, "rep")
    return rep


if __name__ == "__main__":
    overall_train_set, overall_set, alnRes_mat, alnRes_pred_df, cosine_sim_df, cosine_pred_df = get_files()
    pdbs_not_in_full_train = [x for x in alnRes_mat.columns.to_list() if x not in cosine_sim_df.index.to_list()]
    # use the next row if checking the results only on part of the data
    # alnRes_mat = alnRes_mat.drop(pdbs_not_in_full_train, axis=1)
    overall_set.loc[:, "closest"] = overall_set.apply(lambda row: get_label(row, alnRes_mat), axis=1)
    overall_set.loc[:, "highest_from_dif_clust"] = overall_set.apply(lambda row: get_highest_from_dif_clust(row, overall_set, alnRes_mat), axis=1)
    # overall_set = get_closest(overall_set, alnRes_mat)
    # overall_set = pd.read_pickle("/vol/ek/Home/orlyl02/working_dir/oligopred/embed_transfer/overall_set.pkl")
    # overall_set = get_clust_label(overall_set)
    overall_set.to_pickle(PATH + "overall_set_cluster_check_c0.3.pkl")
