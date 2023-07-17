
import os
import pandas as pd
import pickle

class IdentityDict(dict):
    """
    This variant of a dictionary defaults to the identity function if a key has
    no corresponding value. This means that if the key is not found in the dictionary,
    it simply returns the key.
    """
    def __missing__(self, key):
        return key

# The path to the directory that contains the script
root = os.path.dirname(os.path.realpath(__file__))

# Try to load the mapping from a pickle file
try:
    with open(os.path.join(root, "ensembl.pkl"), "rb") as f:
        symbol = pickle.load(f)

# If the pickle file does not exist, create the mapping
except FileNotFoundError:
    # Load the data from a .tsv file
    ensembl = pd.read_csv(os.path.join(root, "ensembl.tsv"), sep="\t")

    # Create an IdentityDict to store the mapping
    symbol = IdentityDict()

    # Populate the dictionary
    for (index, row) in ensembl.iterrows():
        symbol[row["Ensembl ID(supplied by Ensembl)"]] = row["Approved symbol"]

    # Store the mapping as a pickle file for future use
    with open(os.path.join(root, "ensembl.pkl"), "wb") as f:
        pickle.dump(symbol, f)
