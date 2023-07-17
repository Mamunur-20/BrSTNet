import os
import pandas as pd
from PIL import Image
import shutil

Image.MAX_IMAGE_PIXELS = 1000000000 


def change_filename(path, old, new):
    """Change filenames in the given path, replacing old substring with new."""
    fileList = os.listdir(path)
    for oldname in fileList:
        if old in oldname:
            newname = oldname.replace(old, new)
            os.rename(oldname, newname)
            print(oldname, '======>', newname)


def get_key(dict, value):
    """Return keys for the given value in a dictionary."""
    return [k for k, v in dict.items() if value in v]


# First, we need to change all "BT" to "BC" to keep consistency.
change_filename(os.getcwd(), "BT", "BC")

# Load metadata
metadata = pd.read_csv("metadata.csv")

# Create a dictionary with types as keys and patients as values
dictionary = {}
for index, row in metadata.iterrows():
    if row['type'] in dictionary:
        if row['patient'] not in dictionary[row['type']]:
            dictionary[row['type']].append(row['patient'])
    else:
        dictionary[row['type']] = [row['patient']]

# Create directories for subtypes and patients
current_path = os.getcwd()
for st in dictionary.keys():
    os.makedirs(current_path + '/' + st, exist_ok=True)
    for p in dictionary[st]:
        os.makedirs(current_path + '/' + st + '/' + p, exist_ok=True)

# Move files to the corresponding folders
fileList = os.listdir(current_path)
for patient in set(metadata["patient"]):
    patient_st = get_key(dictionary, patient)[0]
    for file in fileList:
        if patient in file:  # get the file name with patient id
            shutil.move(current_path + '/' + file, current_path + '/' + patient_st + '/' + patient)

# Rename files
for root, dirs, files in os.walk(current_path):
    for file in files:
        oldname = os.path.join(root, file)
        if '.jpg' in file:
            newname = oldname.replace("HE_", "")
            os.rename(oldname, newname)
            print(oldname, '======>', newname)
        elif '_stdata' in file:
            newname = oldname.replace("_stdata", "")
            os.rename(oldname, newname)
            print(oldname, '======>', newname)
        elif 'spots_' in file:
            newname = oldname.replace("spots_", "").replace("csv", "spots")
            os.rename(oldname, newname)
            print(oldname, '======>', newname)
