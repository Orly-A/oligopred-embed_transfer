import numpy as np
import pandas as pd

PATH = "/vol/ek/Home/orlyl02/working_dir/oligopred/embed_transfer/"
cov_PATH = "/vol/ek/Home/orlyl02/working_dir/oligopred/embed_transfer/transfer_cov03/"

def parse_alignment(alnRes, col_names):
    alnRes.drop_duplicates(subset=col_names, inplace=True)
    alnRes.sort_values("identity", ascending=False, inplace=True)
    alnRes.drop_duplicates(subset=["query_seq", "target_seq"], keep="first", inplace=True)
    return alnRes


def create_matrix(alnRes):
    new_df = alnRes.pivot(index='query_seq', columns='target_seq', values='identity').reindex(
        index=alnRes["query_seq"].unique(), columns=alnRes["query_seq"].unique()
    ).fillna(0, downcast='infer').astype(pd.SparseDtype("float64", 0))
    # np.max(x - np.eye(*x.shape), axis=0) > 0
    return new_df


def check_duplicates(alnRes):
    """
    use this function to varify that all the remaining duplicates are for cases of query==traget.
    did this for the data, the func is only here if needed to check again
    """
    df1 = alnRes.groupby(["query_seq", "target_seq"]).count().reset_index()
    df2 = df1.loc[df1['identity'] > 1]
    print(df2.target.eq(df2["query_seq"]).value_counts())
    print(df2.shape)
    #these two numbers should be the same


if __name__ == "__main__":
    # I hope these are the correct col names...
    col_names = ["query_seq", "target_seq", "identity", "alnLen", "mismatches",
                 "gapOpens", "qStart", "qEnd", "tStart", "tEnd", "eVal", "bitScore"]
    # alnRes = pd.read_csv(PATH + "fourth_align_res/alnRes.tab", sep="\t", names=col_names)
    # alnRes = pd.read_csv(PATH + "second_align_res/alnRes.tab", sep="\t", names=col_names)
    alnRes = pd.read_csv("/vol/ek/Home/orlyl02/working_dir/oligopred/embed_transfer/blast_ali_eval_100/all-vs-all.tsv", sep="\t", names=col_names)

    alnRes = parse_alignment(alnRes, col_names)
    alnRes_mat = create_matrix(alnRes)
    # save the df results to pkl
    alnRes.to_pickle(PATH + "alnRes_blast_e100_clean.pkl")
    alnRes_mat.to_pickle(PATH + "alnRes_blast_e100_mat.pkl")
    print(alnRes)

