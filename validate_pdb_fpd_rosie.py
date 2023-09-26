import pandas as pd
import biopandas
import numpy as np
import sys
import argparse

from biopandas.pdb import PandasPdb


def get_files():
    """
    Get the files needed from the user
    use the sys.argv module
    -i complex_uploaded.pdb
    -partners <AB_C>
    -o <output filename
    :return: pdb file as bp and chain identification
    """
    #use Argparse in order to have the flag names
    # deal with the parser later
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--name', type=str, required=True)
    # args = parser.parse_args()
    # print(args.name)
    # arg_pars = sys.argv[1:]
    pdb = "4Z2P.pdb"
    chains = "AB_C"
    pdb_name = pdb.rsplit(".", 1)[0]
    ppdb = PandasPdb().fetch_pdb(pdb_name)
    atom_df = ppdb.df['ATOM']
    rec_chain = chains.split("_")[0]
    pep_chain = chains.split("_")[1]

    return atom_df, rec_chain, pep_chain, pdb_name, ppdb


def merged_rec_chains(atom_df, rec_chain):
    """
    Merge the receptor chains into one
    :param atom_df: atom dataframe from the pdb file
    :param rec_chain: receptor chain
    :return: the atom_df with the merged chains
    """

    # check if the chain is a single letter or a list
    if len(rec_chain) > 1:
        if len(rec_chain) > 2:
            print("Warning! The receptor chain is more than 2 chains, if this crashes you should merge all the rec chains manually")
    chain_dict = {}
    for chain in rec_chain:
        chain_range = atom_df[atom_df.chain_id == chain]["residue_number"].unique()
        chain_min = chain_range.min()
        chain_max = chain_range.max()
        chain_dict[chain] = [chain_min, chain_max, chain_range]
    # renumber the residues and rename the chains
    for chain in rec_chain:
        if chain == rec_chain[0]:
            continue
        else:
            chain_dict[chain].append([x for x in chain_dict[chain][2] + chain_dict[rec_chain[0]][1]])
            atom_df.loc[atom_df.chain_id == chain, 'residue_number'] = atom_df[atom_df.chain_id == chain]["residue_number"] + chain_dict[rec_chain[0]][1]
            atom_df.loc[atom_df.chain_id == chain, 'chain_id'] = rec_chain[0]
    with open("naming.txt", "w") as f:
        f.write("chain\tchain_min\tchain_max\tchain_range\trenumbered_range")
        f.write(str(chain_dict))
    return atom_df, chain_dict

def reorder_chains(atom_df, rec_chain, pep_chain):
    """
    Reorder the chains so that the peptide is the last chain
    :return: the atom_df with the reordered chains
    """
    # find the index if the last residue of the rec and make sure it is lower than the first residue of the peptide
    rec_last_res = atom_df[atom_df.chain_id == rec_chain].index.max()
    pep_first_res = atom_df[atom_df.chain_id == pep_chain].index.min()
    if rec_last_res < pep_first_res:
        return atom_df
    else:
        # reorder the indexes
        rec_df = atom_df[atom_df.chain_id == rec_chain]
        pep_df = atom_df[atom_df.chain_id == pep_chain]
        concat_atom_df = pd.concat([rec_df, pep_df], ignore_index=True)

    return concat_atom_df


def check_pep_length(atom_df, pep_chain):
    """
    Check the length of the peptide
    :param atom_df: atom dataframe from the pdb file
    :param pep_chain: peptide chain
    :return: nothing, print warnings or exit if there is a problem
    """
    # count the number of residues in the peptide chain
    pep_length = len(atom_df[atom_df.chain_id == pep_chain]["residue_number"].unique())
    if pep_length > 30:
        print("ERROR! Peptide length is too long, over 30 residues")
        sys.exit()
    elif pep_length < 5:
        print("WARNING! Peptide length is shorter than 5 residues, this is not benchmarked in the protocol but it will run")
    elif pep_length > 15:
        print("WARNING! Peptide length is longer than 15 residues, this is not benchmarked in the protocol but it will run")
    else:
        print("Great! Peptide length is within the benchmarked range")
    return


def check_occupancy(atom_df):
    """
    Check the occupancy of the structure
    we will keep only the highest occupancy or the first rotamer (if are the same)
    Currently only keeping the first rotamer
    :param atom_df: atom dataframe from the pdb file
    :return: the atom_df with corrected occupancy
    """
    atom_df_no_alt_loc = atom_df[atom_df.alt_loc != "A"]
    atom_df_no_alt_loc.loc[:,'occupancy'] = 1
    list_of_alt_res = atom_df[atom_df.occupancy != 1.0].residue_number.unique()
    print("These residues have alternative conformations: ", list_of_alt_res)
    print("We keep only the first rotamer, if you wish to choose a different rotamer, please use a PDB editor of your choice and resumbit the structure")
    # strated to think what to do if we want to look at the rotamers (if the occupancies aren't equal)
    # this isn't finished yet
    partial_occ_df = atom_df[atom_df.occupancy != 1.0]
    for num in partial_occ_df.residue_number.unique():
        current_df = partial_occ_df[partial_occ_df.residue_number == num]
    return atom_df_no_alt_loc




if __name__ == "__main__":
    # open the file
    atom_df, rec_chain, pep_chain, pdb_name, ppdb = get_files()
    if len(rec_chain) > 1:
        atom_df, chain_dict = merged_rec_chains(atom_df, rec_chain)
    # update the rec chain to the new merged chain
    rec_chain = rec_chain[0]
    # reorder the rec and peptide if needed
    atom_df = reorder_chains(atom_df, rec_chain, pep_chain)

    # check the peptide length
    check_pep_length(atom_df, pep_chain)
    # check and fix the occupancy
    atom_df = check_occupancy(atom_df)
    ppdb.df['ATOM'] = atom_df
    # write the new pdb file using biopandas
    ppdb.to_pdb(path=pdb_name + "_fixed.pdb", records=['ATOM'], gz=False, append_newline=True)
    print("If the structure has any PTMs ot hetatoms that are neeeded, please change the type to ATOM manually in the pdb file, otherwise they are not used")
    print("Finished successfully!")

