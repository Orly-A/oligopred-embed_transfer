import pandas as pd
import biopandas
import numpy as np
import sys


def get_files():
    """
    Get the files needed from the user
    use the sys.argv module
    -i complex_uploaded.pdb
    -partners <AB_C>
    -o <output filename
    :return: pdb file as bp and chain identification
    """
    arg_pars = sys.argv[1:]


if __name__ == "__main__":
    pdb, chains = get_files()