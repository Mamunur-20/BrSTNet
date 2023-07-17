# Importing required libraries
import glob
import pathlib
import pickle
import collections
import numpy as np
import openslide
import torch
import torchvision
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 1000000000 

# Local module imports
import utils 

class Generator(torch.utils.data.Dataset): 
    """ This class acts as a Generator for the base-line dataset.
    """
    def __init__(self,
                 patient=None,
                 test_mode=None,
                 window=224,  
                 count_root='data/count_filtered/',
                 img_root='data/image_stained/',
                 count_cached=None,
                 img_cached=None,
                 transform=None):
        """ Initializer/ Constructor for the Generator class
        """
        # Collects the list of all .npz files present in the count_root directory
        self.dataset = sorted(glob.glob("{}*/*/*.npz".format(count_root)))  

        # Filters dataset to include only specified patient data
        if patient is not None:
            self.dataset = [d for d in self.dataset if ((d.split("/")[-2] in patient) or ((d.split("/")[-2], d.split("/")[-1].split("_")[0]) in patient))]
    
        self.test_mode = test_mode
        self.window = window

        self.count_root = count_root
        self.img_root = img_root

        self.count_cached = count_cached
        self.img_cached = img_cached

        self.transform = transform

        # Load subtype: HER2_non_luminal
        with open(self.count_root + "subtype.pkl", "rb") as f:
            self.subtype = pickle.load(f)
     
        self.slide = collections.defaultdict(dict)
        self.slide_mask = collections.defaultdict(dict)
        
        # Load every tif image (can be parallelized)
        for (patient, section) in set([(d.split("/")[-2], d.split("/")[-1].split("_")[0]) for d in self.dataset]):
            self.slide[patient][section] = openslide.open_slide("{}{}/{}/{}_{}.jpg".format(self.img_root, self.subtype[patient], patient, patient, section))
            self.slide_mask[patient][section]  = openslide.open_slide("{}{}/{}/{}_{}_mask.jpg".format(self.img_root, self.subtype[patient], patient, patient, section))

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Gets the item at a specific index in the dataset.

        Args:
            index: The index to retrieve the item from.
        
        Returns:
            Tuple of an image (X) and its corresponding count.
        """
        # Load the file at the current index
        npz = np.load(self.dataset[index])
        # Retrieve the relevant data from the loaded file
        count = npz["count"]
        pixel = npz["pixel"]
        patient = npz["patient"][0]
        section = npz["section"][0]
        coord = npz["index"]

        # Get the relevant slide and mask from the pre-loaded data
        slide = self.slide[patient][section]
        slide_mask = self.slide_mask[patient][section]
        # Read and transform the region of interest from the slide and mask
        X = slide.read_region((pixel[0] - self.window  // 2, pixel[1] - self.window  // 2), 0, (self.window , self.window ))
        X = X.convert("RGB")
        X_mask = slide_mask.read_region((pixel[0] - self.window  // 2, pixel[1] - self.window  // 2), 0, (self.window , self.window ))
        X_mask = X_mask.convert("1")

        # Convert the original and mask images into tensors
        he = X
        he_mask = X_mask
        X = self.transform(X)
        X_mask = self.transform(X_mask)

        # Generate file paths for caching count and image data
        cached_count = "{}{}/{}/{}_{}_{}.npz".format(self.count_cached, self.subtype[patient], patient, section, coord[0], coord[1])
        cached_image = "{}{}/{}/{}/{}_{}_{}.jpg".format(self.img_cached, self.subtype[patient], patient, self.window, section, coord[0], coord[1])

        # Create directories for the cached files if they don't exist
        pathlib.Path(cached_count.strip(cached_count.split('/')[-1])).mkdir(parents=True, exist_ok=True)
        pathlib.Path(cached_image.strip(cached_image.split('/')[-1])).mkdir(parents=True, exist_ok=True)

        # Check if the data is test data
        if self.test_mode == None:
            # Calculate the ratio of white pixels in the mask
            white_ratio = torch.count_nonzero(X_mask * 255) / float(torch.numel(X_mask))
            # If the ratio is less than 0.5, save the count and image data
            if white_ratio < 0.5:
                shutil.copy(self.dataset[index], cached_count)
                he.save(cached_image)
        else:
            # For test data, always save the count and image data
            shutil.copy(self.dataset[index], cached_count)
            he.save(cached_image)
        
        # Return the transformed image and the count
        return X, count


class SubGenerator(torch.utils.data.Dataset): 
    """ This class acts as a Sub-Generator for the ablation study dataset.
    """
    def __init__(self,
                 patient=None,
                 window=224, 
                 resolution=224, 
                 count_root='data/count_filtered/',
                 img_root='data/image_stained/',
                 img_cached=None,
                 transform=None):
        """ Initializer/ Constructor for the SubGenerator class
        """
        # Collects the list of all .npz files present in the count_root directory
        self.dataset = sorted(glob.glob("{}*/*/*.npz".format(count_root)))  

        # Filters dataset to include only specified patient data
        if patient is not None:
            self.dataset = [d for d in self.dataset if ((d.split("/")[-2] in patient) or ((d.split("/")[-2], d.split("/")[-1].split("_")[0]) in patient))]
        
        self.window = window
        self.resolution = resolution

        self.count_root = count_root  # subtype path
        self.img_root = img_root  # training path

        self.img_cached = img_cached  # saved path

        self.transform = transform

        # Load subtype: HER2_non_luminal
        with open(self.count_root + "subtype.pkl", "rb") as f:
            self.subtype = pickle.load(f)
     
        self.slide = collections.defaultdict(dict)
        
        # Load every tif image (can be parallelized)
        for (patient, section) in set([(d.split("/")[-2], d.split("/")[-1].split("_")[0]) for d in self.dataset]):
            self.slide[patient][section] = openslide.open_slide("{}{}/{}/{}_{}.jpg".format(self.img_root, self.subtype[patient], patient, patient, section))

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Fetches an item from the dataset at a specified index.

        Args:
            index: The index to retrieve the item from.
        
        Returns:
            Tuple of an image (X) and its corresponding count.
        """
        # Load the data for the item at the current index
        npz = np.load(self.dataset[index])
        count   = npz["count"]
        pixel   = npz["pixel"]
        patient = npz["patient"][0]
        section = npz["section"][0]
        coord   = npz["index"]  # 11*17

        # Get the relevant slide
        slide = self.slide[patient][section]
        X = slide.read_region((pixel[0] - self.window  // 2, pixel[1] - self.window  // 2), 0, (self.window , self.window ))
        X = X.convert("RGB") 

        # Prepare image for saving
        he = X

        # Transform the image (presumably some sort of normalization or similar)
        X = self.transform(X)

        # If the desired resolution is different from the current resolution, resize the image
        if self.resolution != 224:
            he = torchvision.transforms.Resize((self.resolution, self.resolution))(he)

        # Create the path where the image will be saved, making directories as needed
        cached_image = "{}{}/{}/{}/{}/{}_{}_{}.jpg".format(self.img_cached, self.subtype[patient], patient, self.window, self.resolution, section, coord[0], coord[1])
        pathlib.Path(cached_image.strip(cached_image.split('/')[-1])).mkdir(parents=True, exist_ok=True)

        # Save the image
        he.save(cached_image)

        # Return the transformed image and the count
        return X, count
    

class Spatial(torch.utils.data.Dataset):
    """
    A custom PyTorch Dataset class for spatial gene expression data.
    """

    def __init__(self,
                 patient=None,
                 window=224,
                 resolution=224,
                 count_root=None,
                 img_root=None,
                 gene_filter=250,
                 aux_ratio=1,
                 transform=None,
                 normalization=None,
                 ):

        # The collection of .npz files containing the data
        self.dataset = sorted(glob.glob("{}*/*/*.npz".format(count_root)))

        # Filter the dataset based on the patient argument
        if patient is not None:
            self.dataset = [d for d in self.dataset if ((d.split("/")[-2] in patient) or ((d.split("/")[-2], d.split("/")[-1].split("_")[0]) in patient))]

        self.transform = transform
        self.window = window
        self.resolution = resolution
        self.count_root = count_root
        self.img_root = img_root
        self.gene_filter = gene_filter
        self.aux_ratio = aux_ratio
        self.normalization = normalization

        # Load subtype, gene names and mean expression
        with open("data/count_filtered/subtype.pkl", "rb") as f:
            self.subtype = pickle.load(f)
        with open("data/count_filtered/gene.pkl", "rb") as f:
            self.ensg_names = pickle.load(f)
        self.mean_expression = np.load('data/count_filtered/mean_expression.npy')

        self.gene_names = list(map(lambda x: utils.ensembl.symbol[x], self.ensg_names))

        # Filter genes based on mean expression
        keep_gene = set(list(zip(*sorted(zip(self.mean_expression, range(self.mean_expression.shape[0])))[::-1][:self.gene_filter]))[1])
        self.keep_bool = np.array([i in keep_gene for i in range(len(self.gene_names))])
        self.ensg_keep = [n for (n, f) in zip(self.ensg_names, self.keep_bool) if f]
        self.gene_keep = [n for (n, f) in zip(self.gene_names, self.keep_bool) if f]

        # Create additional auxiliary prediction targets if aux_ratio is not zero
        if self.aux_ratio != 0:
            self.aux_nums = int((len(self.gene_names) - self.gene_filter) * self.aux_ratio)
            aux_gene = set(list(zip(*sorted(zip(self.mean_expression, range(self.mean_expression.shape[0])))[::-1][self.gene_filter:self.gene_filter + self.aux_nums]))[1])
            self.aux_bool = np.array([i in aux_gene for i in range(len(self.gene_names))])
            self.ensg_aux = [n for (n, f) in zip(self.ensg_names, self.aux_bool) if f]
            self.gene_aux = [n for (n, f) in zip(self.gene_names, self.aux_bool) if f]

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Fetches an item from the dataset at a specified index.

        Args:
            index: The index to retrieve the item from.
        
        Returns:
            Tuple containing the image (X), gene expression counts (y), auxiliary counts (aux), 
            coordinate information (coord), patient information, and pixel data.
        """
        npz = np.load(self.dataset[index])

        count = npz["count"]
        pixel = npz["pixel"]
        patient = npz["patient"][0]
        section = npz["section"][0]
        coord = npz["index"]

        # Construct the path to the cached image and load it
        cached_image = "{}{}/{}/{}/{}/{}_{}_{}.jpg".format(self.img_root, self.subtype[patient], patient, self.window, self.resolution, section, coord[0], coord[1])
        X = PIL.Image.open(cached_image)

        # Apply transformations if specified
        if self.transform is not None:
            X = self.transform(X)

        # Resize if necessary
        if X.shape[1] != 224:
            X = torchvision.transforms.Resize((224, 224))(X)

        # Prepare labels
        keep_count = count[self.keep_bool]
        y = torch.as_tensor(keep_count, dtype=torch.float)
        y = torch.log(1 + y)

        # Apply normalization if specified
        if self.normalization is not None:
            y = (y - self.normalization[0]) / self.normalization[1]

        coord = torch.as_tensor(coord)
        index = torch.as_tensor([index])

        if self.aux_ratio != 0:  # Return auxiliary data if required
            aux_count = count[self.aux_bool]
            aux = torch.as_tensor(aux_count, dtype=torch.float)
            aux = torch.log(1 + aux)
            return X, y, aux, coord, index, patient, section, pixel
        else:
            return X, y, coord, index, patient, section, pixel



def get_spatial_patients():
    """
    Function to fetch patient data.

    Returns a dictionary mapping patient names to their corresponding sections.
    """
    patient_section = map(lambda x: x.split("/")[-1].split(".")[0].split("_"), glob.glob("data/image_raw/*/*/*_*.jpg"))
    patient = collections.defaultdict(list)
    for (p, s) in patient_section:
        patient[p].append(s)
    return patient

def patient_or_section(name):
    """
    Function to handle patient data with sections.
    """
    return tuple(name.split("_")) if "_" in name else name

def get_sections(patients, testpatients):
    """
    Function to segregate train and test patients.
    """
    train_patients = []
    test_patients = []
    for (i, p) in enumerate(patients):
        for s in patients[p]:
            if p in testpatients:
                test_patients.append((p, s))
            else:
                train_patients.append((p, s))
    
    print('Train patients: ',train_patients)
    print('Test patients: ', test_patients)

    return train_patients, test_patients

def cv_split(patients, cv_folds):
    """
    Function to perform cross-validation split.
    """
    fold = [patients[f::cv_folds] for f in range(cv_folds)]
    for f in range(cv_folds):
        print("Fold #{}".format(f))
        train = [fold[i] for i in range(5) if i != f]
        train = [i for sublist in train for i in sublist]  # flatten
        test = fold[f]
    return train, test

class LRScheduler():
    """
    Learning rate scheduler.
    """
    def __init__(self, optimizer, patience=5, min_lr=1e-6, factor=0.1):
        """
        new_lr = old_lr * factor
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
            )

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=20, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

# 'util.py' may have been included here in the original script

# 'generator.py' starts here

dataset = sorted(glob.glob("data/count_filtered/*/*/*.npz"))  # collection of .npz files

patients = sorted(get_spatial_patients().keys())
test_patients = ["BC23903"]
train_patients = [p for p in patients if p not in test_patients]

print("Train patients: ",  train_patients)
print("Test patients: ", test_patients)
print()  

# Filter dataset to include only training patients
dataset = [d for d in dataset if ((d.split("/")[-2] in train_patients) or ((d.split("/")[-2], d.split("/")[-1].split("_")[0]) in train_patients))]
    
train_dataset = Generator(train_patients,
                        count_cached = 'training/counts/',
                        img_cached = 'training/images/',
                        transform=torchvision.transforms.ToTensor())  # transform range [0, 255] to [0.0,1.0]

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, 
                        num_workers=0, shuffle=True)

test_dataset = Generator(test_patients,
                        test_mode = 1,
                        count_cached = 'test/counts/',
                        img_cached = 'test/images/',
                        transform=torchvision.transforms.ToTensor())  # transform range [0, 255] to [0.0,1.0]

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, 
                        num_workers=0, shuffle=True)

# Save filtered images and counts
for (i, (he, npz)) in enumerate(train_loader):
    print(f"Saving filtered images and counts: {(i + 1) * 32}/{len(train_dataset)}...")

for (i, (he, npz)) in enumerate(test_loader):
    print(f"Saving filtered images and counts: {(i + 1) * 32}/{len(test_dataset)}...")


