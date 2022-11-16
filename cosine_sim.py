import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


PATH = "/vol/ek/Home/orlyl02/working_dir/oligopred/embed_transfer/"
cov_PATH = "/vol/ek/Home/orlyl02/working_dir/oligopred/embed_transfer/transfer_cov03/"


def get_cosine_sim(train_set):
    cosine_sim = cosine_similarity(train_set['embeddings'].tolist(), train_set["embeddings"].tolist())
    cosine_sim_df = pd.DataFrame(cosine_sim)
    cosine_sim_df.set_axis(train_set["code"].to_list(), axis=1, inplace=True)
    cosine_sim_df.set_axis(train_set["code"].to_list(), axis=0, inplace=True)
    return cosine_sim_df


if __name__ == "__main__":
    # train_set = pd.read_pickle(PATH + "train_set.pkl")
    train_set = pd.read_pickle(cov_PATH + "train_set_c0.3.pkl")
    cosine_sim_df = get_cosine_sim(train_set)
    # cosine_sim_df.to_pickle(PATH + "cosine_sim_df.pkl")
    cosine_sim_df.to_pickle(cov_PATH + "cosine_sim_df_c0.3.pkl")
    print(train_set)
