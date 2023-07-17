# Import necessary libraries
import os
import numpy as np
import pickle
import logging
import pathlib
import time
import glob
import collections
import tqdm
import argparse
import PIL
from PIL import Image
import skimage
import shutil

# Set maximum number of pixels for an image
PIL.Image.MAX_IMAGE_PIXELS = 1000000000    

# Function to generate the count file for single spot
def spatial(args):
    window = 224  # only to check if patch is off of boundary

    logger = logging.getLogger(__name__)

    # Create directory for processed files
    pathlib.Path(args.process).mkdir(parents=True, exist_ok=True)

    # Load raw data and subtype
    raw, subtype = load_raw(args.root)

    # Save subtype
    with open(args.process + "subtype.pkl", "wb") as f:   
        pickle.dump(subtype, f)

    t = time.time()
    t0 = time.time()

    section_header = None
    gene_names = set()

    # Get all gene names
    for patient in raw:
        for section in raw[patient]:
            section_header = raw[patient][section]["count"].columns.values[0]  # Unnamed: 0
            gene_names = gene_names.union(set(raw[patient][section]["count"].columns.values[1:]))
    gene_names = list(gene_names) # without header
    gene_names.sort()  # sort by name

    # Save all sorted gene names
    with open(args.process + "gene.pkl", "wb") as f:     
        pickle.dump(gene_names, f)

    print("Finding list of genes: " + str(time.time() - t0))

    # Process each patient
    for (i, patient) in enumerate(raw):
        print("Processing " + str(i + 1) + " / " + str(len(raw)) + ": " + patient)

        # Create directory for each patient
        pathlib.Path("{}{}/{}/".format(args.process, subtype[patient], patient)).mkdir(parents=True, exist_ok=True)

        # Process each section
        for section in raw[patient]:
            print("Processing " + patient + " " + section + "...")

            # Add zeros to missing gene counts and order columns
            t0 = time.time()
            missing = list(set(gene_names) - set(raw[patient][section]["count"].keys())) # missing names without header
            c = raw[patient][section]["count"].values[:, 1:].astype(float)  # without header
            pad = np.zeros((c.shape[0], len(missing)))
            c = np.concatenate((c, pad), axis=1)
            names = np.concatenate((raw[patient][section]["count"].keys().values[1:], np.array(missing)))  # raw name + missing name
            c = c[:, np.argsort(names)]  # return index and sort (no header)
            print("Adding zeros and ordering columns: " + str(time.time() - t0))

            # Extract counts
            t0 = time.time()
            count = {}      
            for (j, row) in raw[patient][section]["count"].iterrows():  
                count[row.values[0]] = c[j, :]   
            print("Extracting counts: " + str(time.time() - t0))

            # Load image
            image = skimage.io.imread(raw[patient][section]["image"])
            print("Loading image: " + str(time.time() - t0))

            # Process each spot
            for (_, row) in raw[patient][section]["spot"].iterrows():
                x = round(float(row[0].split(',')[1]))   # coord: float (4622, 4621)
                y = round(float(row[0].split(',')[2]))
                spot_x = row[0].split(",")[0].split('x')[0] # spot id 11x17  str
                spot_y = row[0].split(",")[0].split('x')[1] 

                # Check if patch is off of boundary and if spot has gene expression data
                if (x + (-window // 2))>= 0 and (x + (window // 2)) <= image.shape[1] and (y + (-window // 2))>= 0 and (y + (window // 2)) <= image.shape[0]:
                    if (spot_x + "x" + spot_y) in list(count.keys()) :
                        # Check if total read counts is more than 1000
                        if np.sum(count[spot_x + "x" + spot_y]) >= 1000: 
                            # Save data for each spot
                            filename = "{}{}/{}/{}_{}_{}.npz".format(args.process, subtype[patient], patient, section, spot_x, spot_y)
                            np.savez_compressed(filename, 
                                                count=count[spot_x + "x" + spot_y],
                                                pixel=np.array([x, y]),
                                                patient=np.array([patient]),
                                                section=np.array([section]),
                                                index=np.array([int(spot_x), int(spot_y)]))
                        else:
                            logger.warning("Total counts of Patch " + str(spot_x) + "x" + str(spot_y) + " in " + patient + " " + section + " less than 1500")
                    else:
                        logger.warning("Patch " + str(spot_x) + "x" + str(spot_y) + " not found in " + patient + " " + section)
                else:
                    logger.warning("Detected " + patient +  " " + section + " " + str(spot_x) + "x" + str(spot_y) + " too close to edge.")

            print("Saving patches: " + str(time.time() - t0))

    print("Preprocessing took " + str(time.time() - t) + " seconds")

    # Compute and save mean gene expression
    logging.info("Computing statistics of dataset")   
    gene = []           
    for filename in tqdm.tqdm(glob.glob("{}*/*/*_*_*.npz".format(args.process))):
        npz = np.load(filename)
        count = npz["count"]   
        gene.append(np.expand_dims(count, 1))
    gene = np.concatenate(gene, 1)  
    print( "There are {} genes and {} spots left before filtering.".format(gene.shape[0], gene.shape[1]))
    np.save(args.process + "mean_expression.npy", np.mean(gene, 1))

    # Filter genes with zero mean expression and expressed in less than 10% of the array spots
    filter = np.where(np.sum(np.where(gene > 0, 1, 0), 1) >= 0.10 * gene.shape[1], True, False)

    # Update all the files
    print(gene[filter].shape)  #(5943, 30625)

    # Create directory for filtered data
    pathlib.Path(args.filter).mkdir(parents=True, exist_ok=True)

    # Save subtype and gene names
    with open(args.filter + "subtype.pkl", "wb") as f:   
        pickle.dump(subtype, f)
    with open(args.filter + "gene.pkl", "wb") as f:     
        gene_names = list(np.array(gene_noheader)[filter])
        pickle.dump(gene_names, f)

    # Update counts
    for filename in tqdm.tqdm(glob.glob("{}*/*/*_*_*.npz".format(args.process))):
        new_path = "{}{}/{}/".format(args.filter, filename.split('/')[2], filename.split('/')[3])
        pathlib.Path(new_path).mkdir(parents=True, exist_ok=True)
        npz = np.load(filename)
        new_filename = filename.replace(args.process, args.filter)
        np.savez_compressed(new_filename, 
                            count = npz["count"][filter],                                
                            pixel= npz["pixel"],
                            patient=npz["patient"],
                            section=npz["section"],
                            index= npz["index"])

    # Compute and save mean gene expression for filtered data
    logging.info("Computing statistics of dataset")   
    gene = []
    for filename in tqdm.tqdm(glob.glob("{}*/*/*_*_*.npz".format(args.filter))):
        npz = np.load(filename)
        count = npz["count"]
        gene.append(np.expand_dims(count, 1))
    gene = np.concatenate(gene, 1)  
    print( "There are {} genes and {} spots left after filtering.".format(gene.shape[0], gene.shape[1]))
    np.save(args.filter + "mean_expression.npy", np.mean(gene, 1))

# Function to check if file1 is newer than file2
def newer_than(file1, file2):
    return os.path.isfile(file1) and (not os.path.isfile(file2) or os.path.getctime(file1) > os.path.getctime(file2))

# Function to load data for one section of a patient
def load_section(root: str, patient: str, section: str, subtype: str):
    import pandas
    import gzip

    file_root = root + subtype + "/" + patient + "/" + patient + "_" + section
    image = file_root + ".jpg"
    with gzip.open(file_root + ".tsv.gz", "rb") as f:
        count = pandas.read_csv(f, sep="\t")
    spot = pandas.read_csv(file_root + ".spots.gz", sep="\t")
    return {"image": image, "count": count, "spot": spot}

# Function to load data for all patients
def load_raw(root: str):
    images = glob.glob(root + "*/*/*_*.jpg")
    patient = collections.defaultdict(list)
    for (p, s) in map(lambda x: x.split("/")[-1][:-4].split("_"), images):
        patient[p].append(s)
    subtype = {}
    for (st, p) in map(lambda x: (x.split("/")[2], x.split("/")[3]), images):
            subtype[p] = st
    print("Loading raw data...")
    t = time.time()
    data = {}
    with tqdm.tqdm(total=sum(map(len, patient.values()))) as pbar:
        for p in patient:
            data[p] = {}
            for s in patient[p]:
                data[p][s] = load_section(root, p, s, subtype[p])
                pbar.update()
    print("Loading raw data took " + str(time.time() - t) + " seconds.")
    return data, subtype

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate the necessary files.')
parser.add_argument('--root',  type=str, default='data/STBC/', help='Path for the raw dataset')     
parser.add_argument('--process',  type=str, default='data/count_raw/', help='Path for the generated files')
parser.add_argument('--filter',  type=str, default='data/count_filtered/', help='Path for the filtered dataset')
args = parser.parse_args()

# Run spatial function
spatial(args)
