import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


# PATH = "/vol/ek/Home/orlyl02/working_dir/oligopred/embed_transfer/"

# cov_PATH = "/vol/ek/Home/orlyl02/working_dir/oligopred/embed_transfer/transfer_cov03/"
cov_PATH = "/vol/ek/Home/orlyl02/working_dir/oligopred/embed_transfer/esm_embeds/"

def get_cosine_sim(train_set):
    cosine_sim = cosine_similarity(train_set['esm_embeddings'].tolist(), train_set["esm_embeddings"].tolist())
    cosine_sim_df = pd.DataFrame(cosine_sim)
    cosine_sim_df.set_axis(train_set["code"].to_list(), axis=1, inplace=True)
    cosine_sim_df.set_axis(train_set["code"].to_list(), axis=0, inplace=True)
    return cosine_sim_df


def get_cosine_sim_holdout(train_set, holdout_set):
    cosine_sim = cosine_similarity(holdout_set['esm_embeddings'].tolist(), train_set["esm_embeddings"].tolist())
    cosine_sim_df = pd.DataFrame(cosine_sim)
    cosine_sim_df.set_axis(train_set["code"].to_list(), axis=1, inplace=True)
    cosine_sim_df.set_axis(holdout_set["code"].to_list(), axis=0, inplace=True)
    return cosine_sim_df



if __name__ == "__main__":
    # train_set = pd.read_pickle(PATH + "train_set.pkl")
    # train_set = pd.read_pickle(cov_PATH + "train_set_c0.3.pkl")
    # for transfer with esm
    train_set = pd.read_pickle(
        "/vol/ek/Home/orlyl02/working_dir/oligopred/esmfold_prediction/train_set_c0.3.pkl")
    holdout_set = pd.read_pickle(
        "/vol/ek/Home/orlyl02/working_dir/oligopred/esmfold_prediction/hold_out_set_c0.3.pkl")
    cosine_sim_df = get_cosine_sim(train_set)
    holdout_cos_df = get_cosine_sim_holdout(train_set, holdout_set)
    # cosine_sim_df.to_pickle(PATH + "cosine_sim_df.pkl")
    # cosine_sim_df.to_pickle(cov_PATH + "cosine_sim_df_c0.3_esm.pkl")
    holdout_cos_df.to_pickle(cov_PATH + "cosine_sim_df_holdout_c0.3_esm.pkl")
    print(train_set)
