
import torch
import random
import argparse
import math
import torchvision
import utils
import pathlib
import numpy as np
import os
from sklearn import metrics
import pandas
import pickle
import collections
import ensembl
#### metric.py
### R2, RMSE, RMAE, CC

import numpy as np
import torch

def average_correlation_coefficient(y_pred, y_true):
    """
    Calculate Average Correlation Coefficient
    Args:
        y_true (array-like): np.ndarray or torch.Tensor of dimension N x d with actual values
        y_pred (array-like): np.ndarray or torch.Tensor of dimension N x d with predicted values
    Returns:
        float: Average Correlation Coefficient 
    Raises: 
        ValueError: If Parameters are not both of type np.ndarray or torch.Tensor 
    """
    if isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray):
        top = np.sum((y_true - np.mean(y_true, axis=0)) *
                     (y_pred - np.mean(y_pred, axis=0)), axis=0)
        bottom = np.sqrt(np.sum((y_true - np.mean(y_true, axis=0))**2, axis=0) *
                         np.sum((y_pred - np.mean(y_pred, axis=0))**2, axis=0))
        return np.sum(top / bottom) / len(y_true[0])

    elif isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor):
        top = torch.sum((y_true - torch.mean(y_true, dim=0))
                        * (y_pred - torch.mean(y_pred, dim=0)), dim=0)
        bottom = torch.sqrt(torch.sum((y_true - torch.mean(y_true, dim=0))**2, dim=0) *
                            torch.sum((y_pred - torch.mean(y_pred, dim=0))**2, dim=0))
        return torch.sum(top / bottom) / len(y_true[0])

    else:
        raise ValueError('y_true and y_pred must be both of type numpy.ndarray or torch.Tensor')

### Training Loop ###
def fit(model, train_loader, optim, criterion, args, device):
    """
    Training loop for the model
    Args:
        model: The model to be trained
        train_loader: DataLoader for the training data
        optim: The optimizer for updating model parameters
        criterion: The loss function
        args: Additional arguments
        device: Device on which to perform the training
    Returns:
        Tuple: (total_loss, total_aMAE, total_aRMSE, total_aCC) or (total_main_loss, main_aMAE, main_aRMSE, main_aCC)
    """
    print('-' * 10)
    print('Training:')
    model.train()
    total_loss = 0
    total_main_loss = 0
    total_aux_loss = 0

    epoch_count = []
    epoch_preds = []
    aux_count = []
    aux_preds = []

    if args.aux_ratio != 0:  # Return auxiliary, revise model and loss
        for (i, (X, y, aux, c, ind, pat, s, pix)) in enumerate(train_loader):
            X, y, aux = X.to(device), y.to(device), aux.to(device)
            pred = model(X)

            epoch_count.append(y.cpu().detach().numpy())
            epoch_preds.append(pred[0].cpu().detach().numpy())
            aux_count.append(aux.cpu().detach().numpy())
            aux_preds.append(pred[1].cpu().detach().numpy())

            optim.zero_grad()

            main_loss = criterion(pred[0], y)
            aux_loss = criterion(pred[1], aux)
            loss = main_loss + args.aux_weight * aux_loss
            total_loss += loss.cpu().detach().numpy()
            total_main_loss += main_loss.cpu().detach().numpy()
            total_aux_loss += aux_loss.cpu().detach().numpy()

            if args.debug and i == 3:
                break

            loss.backward()
            optim.step()

        total_loss /= len(train_loader)
        total_main_loss /= len(train_loader)
        total_aux_loss /= len(train_loader)

        epoch_count = np.concatenate(epoch_count)
        epoch_preds = np.concatenate(epoch_preds)
        aux_count = np.concatenate(aux_count)
        aux_preds = np.concatenate(aux_preds)

        main_aMAE = metrics.mean_absolute_error(epoch_count, epoch_preds)
        main_aRMSE = metrics.mean_squared_error(epoch_count, epoch_preds, squared=False)
        main_aCC = average_correlation_coefficient(epoch_preds, epoch_count)

        aux_aMAE = metrics.mean_absolute_error(aux_count, aux_preds)
        aux_aRMSE = metrics.mean_squared_error(aux_count, aux_preds, squared=False)
        aux_aCC = average_correlation_coefficient(aux_preds, aux_count)

        print("Total: Loss = {:.4f};".format(total_loss))
        print("Main : Loss = {:.4f}, aMAE = {:.4f}, aRMSE = {:.4f}, aCC = {:.4f};".format(
            total_main_loss, main_aMAE, main_aRMSE, main_aCC))
        print("Aux  : Loss = {:.4f}, aMAE = {:.4f}, aRMSE = {:.4f}, aCC = {:.4f};".format(
            total_aux_loss, aux_aMAE, aux_aRMSE, aux_aCC))

        return total_main_loss, main_aMAE, main_aRMSE, main_aCC

    else:
        for (i, (X, y, c, ind, pat, s, pix)) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            pred = model(X)

            epoch_count.append(y.cpu().detach().numpy())
            epoch_preds.append(pred.cpu().detach().numpy())

            optim.zero_grad()

            loss = criterion(pred, y)

            total_loss += loss.cpu().detach().numpy()

            if args.debug and i == 3:
                break

            loss.backward()
            optim.step()

        total_loss /= len(train_loader)
        epoch_count = np.concatenate(epoch_count)
        epoch_preds = np.concatenate(epoch_preds)

        total_aMAE = metrics.mean_absolute_error(epoch_count, epoch_preds)
        total_aRMSE = metrics.mean_squared_error(epoch_count, epoch_preds, squared=False)
        total_aCC = average_correlation_coefficient(epoch_preds, epoch_count)

        print("Loss = {:.4f}, aMAE = {:.4f}, aRMSE = {:.4f}, aCC = {:.4f}".format(
            total_loss, total_aMAE, total_aRMSE, total_aCC))

        return total_loss, total_aMAE, total_aRMSE, total_aCC


def validate(model, val_loader, criterion, args, device):
    """
    Validation loop for the model
    Args:
        model: The model to be validated
        val_loader: DataLoader for the validation data
        criterion: The loss function
        args: Additional arguments
        device: Device on which to perform the validation
    Returns:
        Tuple: (total_loss, total_aMAE, total_aRMSE, total_aCC) or (total_main_loss, main_aMAE, main_aRMSE, main_aCC)
    """
    print('Validate:')
    model.eval()
    total_loss = 0
    total_main_loss = 0
    total_aux_loss = 0
    epoch_count = []
    epoch_preds = []
    aux_count = []
    aux_preds = []

    with torch.no_grad():
        if args.aux_ratio != 0:  # Return auxiliary, revise model and loss
            for (i, (X, y, aux, c, ind, pat, s, pix)) in enumerate(val_loader):
                X, y, aux = X.to(device), y.to(device), aux.to(device)
                pred = model(X)

                epoch_count.append(y.cpu().detach().numpy())
                epoch_preds.append(pred[0].cpu().detach().numpy())
                aux_count.append(aux.cpu().detach().numpy())
                aux_preds.append(pred[1].cpu().detach().numpy())

                main_loss = criterion(pred[0], y)  # batch-gene average
                aux_loss = criterion(pred[1], aux)
                loss = main_loss + args.aux_weight * aux_loss
                total_loss += loss.cpu().detach().numpy()
                total_main_loss += main_loss.cpu().detach().numpy()
                total_aux_loss += aux_loss.cpu().detach().numpy()

                if args.debug and i == 3:
                    break

            total_loss /= len(val_loader)
            total_main_loss /= len(val_loader)
            total_aux_loss /= len(val_loader)

            epoch_count = np.concatenate(epoch_count)
            epoch_preds = np.concatenate(epoch_preds)
            aux_count = np.concatenate(aux_count)
            aux_preds = np.concatenate(aux_preds)

            main_aMAE = metrics.mean_absolute_error(epoch_count, epoch_preds)
            main_aRMSE = metrics.mean_squared_error(epoch_count, epoch_preds, squared=False)
            main_aCC = average_correlation_coefficient(epoch_preds, epoch_count)

            aux_aMAE = metrics.mean_absolute_error(aux_count, aux_preds)
            aux_aRMSE = metrics.mean_squared_error(aux_count, aux_preds, squared=False)
            aux_aCC = average_correlation_coefficient(aux_preds, aux_count)

            print("Total: Loss = {:.4f};".format(total_loss))
            print("Main : Loss = {:.4f}, aMAE = {:.4f}, aRMSE = {:.4f}, aCC = {:.4f};".format(
                total_main_loss, main_aMAE, main_aRMSE, main_aCC))
            print("Aux  : Loss = {:.4f}, aMAE = {:.4f}, aRMSE = {:.4f}, aCC = {:.4f};".format(
                total_aux_loss, aux_aMAE, aux_aRMSE, aux_aCC))

            return total_main_loss, main_aMAE, main_aRMSE, main_aCC
        else:
            for (i, (X, y, c, ind, pat, s, pix)) in enumerate(val_loader):
                X, y = X.to(device), y.to(device)
                pred = model(X)

                epoch_count.append(y.cpu().detach().numpy())
                epoch_preds.append(pred.cpu().detach().numpy())
                loss = criterion(pred, y)  # batch-gene average
                total_loss += loss.cpu().detach().numpy()

                if args.debug and i == 3:
                    break

            total_loss /= len(val_loader)
            epoch_count = np.concatenate(epoch_count)
            epoch_preds = np.concatenate(epoch_preds)

            total_aMAE = metrics.mean_absolute_error(epoch_count, epoch_preds)
            total_aRMSE = metrics.mean_squared_error(epoch_count, epoch_preds, squared=False)
            total_aCC = average_correlation_coefficient(epoch_preds, epoch_count)

            print("Loss = {:.4f}, aMAE = {:.4f}, aRMSE = {:.4f}, aCC = {:.4f}".format(
                total_loss, total_aMAE, total_aRMSE, total_aCC))

            return total_loss, total_aMAE, total_aRMSE, total_aCC


def test(model, test_loader, criterion, device, args, epoch):
    """
    Test loop for the model
    Args:
        model: The model to be tested
        test_loader: DataLoader for the test data
        criterion: The loss function
        device: Device on which to perform the testing
        args: Additional arguments
        epoch: Current epoch
    Returns:
        Tuple: (total_loss, total_aMAE, total_aRMSE, total_aCC) or (total_main_loss, main_aMAE, main_aRMSE, main_aCC)
    """
    print('Test:')
    model.eval()
    total_loss = 0
    total_main_loss = 0
    total_aux_loss = 0

    epoch_count = []
    epoch_preds = []
    aux_count = []
    aux_preds = []
    patient = []
    section = []
    coord = []
    pixel = []

    with torch.no_grad():
        if args.aux_ratio != 0:  # Return auxiliary, revise model and loss
            for (i, (X, y, aux, c, ind, pat, s, pix)) in enumerate(test_loader):
                X, y, aux = X.to(device), y.to(device), aux.to(device)
                pred = model(X)

                epoch_count.append(y.cpu().detach().numpy())
                epoch_preds.append(pred[0].cpu().detach().numpy())

                aux_count.append(aux.cpu().detach().numpy())
                aux_preds.append(pred[1].cpu().detach().numpy())

                main_loss = criterion(pred[0], y)  # batch-gene average
                aux_loss = criterion(pred[1], aux)
                loss = main_loss + args.aux_weight * aux_loss
                total_loss += loss.cpu().detach().numpy()
                total_main_loss += main_loss.cpu().detach().numpy()
                total_aux_loss += aux_loss.cpu().detach().numpy()

                if args.debug and i == 3:
                    break

                patient += pat
                section += s
                coord.append(c.detach().numpy())
                pixel.append(pix.detach().numpy())

            total_loss /= len(test_loader)
            total_main_loss /= len(test_loader)
            total_aux_loss /= len(test_loader)

            epoch_count = np.concatenate(epoch_count)
            epoch_preds = np.concatenate(epoch_preds)

            aux_count = np.concatenate(aux_count)
            aux_preds = np.concatenate(aux_preds)

            main_aMAE = metrics.mean_absolute_error(epoch_count, epoch_preds)
            main_aRMSE = metrics.mean_squared_error(epoch_count, epoch_preds, squared=False)
            main_aCC = average_correlation_coefficient(epoch_preds, epoch_count)

            aux_aMAE = metrics.mean_absolute_error(aux_count, aux_preds)
            aux_aRMSE = metrics.mean_squared_error(aux_count, aux_preds, squared=False)
            aux_aCC = average_correlation_coefficient(aux_preds, aux_count)

            print("Total: Loss = {:.4f};".format(total_loss))
            print("Main : Loss = {:.4f}, aMAE = {:.4f}, aRMSE = {:.4f}, aCC = {:.4f};".format(
                total_main_loss, main_aMAE, main_aRMSE, main_aCC))
            print("Aux  : Loss = {:.4f}, aMAE = {:.4f}, aRMSE = {:.4f}, aCC = {:.4f};".format(
                total_aux_loss, aux_aMAE, aux_aRMSE, aux_aCC))

            coord = np.concatenate(coord)
            pixel = np.concatenate(pixel)

            if args.pred_root:
                pathlib.Path(os.path.dirname(args.pred_root + '/')).mkdir(parents=True, exist_ok=True)
                np.savez_compressed(args.pred_root + "/epoch_" + str(epoch),  # stopped epoch + 1 - patience
                                    task="gene",
                                    counts=epoch_count,
                                    predictions=epoch_preds,
                                    aux_counts=aux_count,
                                    aux_predictions=aux_preds,
                                    coord=coord,
                                    patient=patient,
                                    section=section,
                                    pixel=pixel,
                                    ensg_names=test_loader.dataset.ensg_keep,
                                    gene_names=test_loader.dataset.gene_keep,
                                    aux_ensg_names=test_loader.dataset.ensg_aux,
                                    aux_gene_names=test_loader.dataset.gene_aux,
                                    )

            return total_main_loss, main_aMAE, main_aRMSE, main_aCC
        else:
            for (i, (X, y, c, ind, pat, s, pix)) in enumerate(test_loader):
                X, y = X.to(device), y.to(device)
                pred = model(X)

                epoch_count.append(y.cpu().detach().numpy())
                epoch_preds.append(pred.cpu().detach().numpy())

                loss = criterion(pred, y)  # batch-gene average
                total_loss += loss.cpu().detach().numpy()

                if args.debug and i == 3:
                    break

                patient += pat
                section += s
                coord.append(c.detach().numpy())
                pixel.append(pix.detach().numpy())

            total_loss /= len(test_loader)
            epoch_count = np.concatenate(epoch_count)
            epoch_preds = np.concatenate(epoch_preds)

            total_aMAE = metrics.mean_absolute_error(epoch_count, epoch_preds)
            total_aRMSE = metrics.mean_squared_error(epoch_count, epoch_preds, squared=False)
            total_aCC = average_correlation_coefficient(epoch_preds, epoch_count)

            print("Loss = {:.4f}, aMAE = {:.4f}, aRMSE = {:.4f}, aCC = {:.4f}".format(
                total_loss, total_aMAE, total_aRMSE, total_aCC))

            coord = np.concatenate(coord)
            pixel = np.concatenate(pixel)

            if args.pred_root:
                pathlib.Path(os.path.dirname(args.pred_root + '/')).mkdir(parents=True, exist_ok=True)
                np.savez_compressed(args.pred_root + "/epoch_" + str(epoch),  # stopped epoch + 1 - patience
                                    task="gene",
                                    counts=epoch_count,
                                    predictions=epoch_preds,
                                    coord=coord,
                                    patient=patient,
                                    section=section,
                                    pixel=pixel,
                                    ensg_names=test_loader.dataset.ensg_keep,
                                    gene_names=test_loader.dataset.gene_keep,
                                    )

            return total_loss, total_aMAE, total_aRMSE, total_aCC


class IdentityDict(dict):
    """This variant of a dict defaults to the identity function if a key has
    no corresponding value.

    https://stackoverflow.com/questions/6229073/how-to-make-a-python-dictionary-that-returns-key-for-keys-missing-from-the-dicti
    """
    def __missing__(self, key):
        return key

root = os.path.dirname(os.path.realpath("__file__"))
try:
    with open(os.path.join(root, "ensembl.pkl"), "rb") as f:
        symbol = pickle.load(f)
except FileNotFoundError:
    ensembl = pandas.read_csv(os.path.join(root, "ensembl.tsv"), sep="\t")

    # TODO: should just return Ensembl ID if no name available
    symbol = IdentityDict()

    for (index, row) in ensembl.iterrows():
        symbol[row["Ensembl ID(supplied by Ensembl)"]] = row["Approved symbol"]

    with open(os.path.join(root, "ensembl.pkl"), "wb") as f:
        pickle.dump(symbol, f)



def get_mean_and_std(loader, args):
    t = time.time()
    mean = 0.
    std = 0.
    nb_samples = 0
    epoch_count = []

    for (i, (X, y, *_)) in enumerate(loader):
        if args.debug and i == 3:
            break

        batch_samples = X.size(0)
        X = X.view(batch_samples, X.size(1), -1)
        mean += X.mean(2).sum(0)
        std += X.std(2).sum(0)
        nb_samples += batch_samples
        epoch_count.append(y)

    mean /= nb_samples
    std /= nb_samples
    epoch_count = torch.cat(epoch_count, dim=0)
    mean_count = epoch_count.mean(0)
    std_count = epoch_count.std(0)

    print("Computing mean and std of gene expressions, estimating mean ({:.4f}) and std ({:.4f}) took {:.4f}s".format(
        mean, std, time.time() - t))
    print()

    return mean.tolist(), std.tolist(), mean_count, std_count


class Generator(torch.utils.data.Dataset):  # for baseline
    def __init__(self,
                 patient=None,
                 test_mode=None,
                 window=224,
                 count_root='data/count_filtered/',
                 img_root='data/image_stained/',
                 count_cached=None,
                 img_cached=None,
                 transform=None,
                 ):
        self.dataset = sorted(glob.glob("{}*/*/*.npz".format(count_root)))

        if patient is not None:
            self.dataset = [d for d in self.dataset if ((d.split("/")[-2] in patient) or ((d.split("/")[-2], d.split("/")[-1].split("_")[0]) in patient))]

        self.test_mode = test_mode
        self.window = window

        self.count_root = count_root
        self.img_root = img_root

        self.count_cached = count_cached
        self.img_cached = img_cached

        self.transform = transform

        with open(self.count_root + "subtype.pkl", "rb") as f:
            self.subtype = pickle.load(f)

        self.slide = collections.defaultdict(dict)
        self.slide_mask = collections.defaultdict(dict)

        for (patient, section) in set([(d.split("/")[-2], d.split("/")[-1].split("_")[0]) for d in self.dataset]):
            self.slide[patient][section] = openslide.open_slide("{}{}/{}/{}_{}.jpg".format(self.img_root, self.subtype[patient], patient, patient, section))
            self.slide_mask[patient][section]  = openslide.open_slide("{}{}/{}/{}_{}_mask.jpg".format(self.img_root, self.subtype[patient], patient, patient, section))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        npz = np.load(self.dataset[index])
        count = npz["count"]
        pixel = npz["pixel"]
        patient = npz["patient"][0]
        section = npz["section"][0]
        coord = npz["index"]

        slide = self.slide[patient][section]
        slide_mask = self.slide_mask[patient][section]
        X = slide.read_region((pixel[0] - self.window  // 2, pixel[1] - self.window  // 2), 0, (self.window , self.window ))
        X = X.convert("RGB")
        X_mask = slide_mask.read_region((pixel[0] - self.window  // 2, pixel[1] - self.window  // 2), 0, (self.window , self.window ))
        X_mask = X_mask.convert("1")

        he = X
        he_mask = X_mask
        X = self.transform(X)
        X_mask = self.transform(X_mask)

        cached_count = "{}{}/{}/{}_{}_{}.npz".format(self.count_cached, self.subtype[patient], patient, section, coord[0], coord[1])
        cached_image = "{}{}/{}/{}/{}_{}_{}.jpg".format(self.img_cached, self.subtype[patient], patient, self.window, section, coord[0], coord[1])
        cached_image_mask = "{}{}/{}/{}/{}_{}_{}_mask.jpg".format(self.img_cached, self.subtype[patient], patient, self.window, section, coord[0], coord[1])

        pathlib.Path(cached_count.strip(cached_count.split('/')[-1])).mkdir(parents=True, exist_ok=True)
        pathlib.Path(cached_image.strip(cached_image.split('/')[-1])).mkdir(parents=True, exist_ok=True)

        if self.test_mode == None:
            white_ratio = torch.count_nonzero(X_mask * 255) / float(torch.numel(X_mask))

            if white_ratio < 0.5:
                shutil.copy(self.dataset[index], cached_count)
                he.save(cached_image)

        else:
            shutil.copy(self.dataset[index], cached_count)
            he.save(cached_image)

        return X, count

class SubGenerator(torch.utils.data.Dataset):  # for ablation
    def __init__(self,
                 patient=None,
                 window=224,
                 resolution=224,
                 count_root='data/count_filtered/',
                 img_root='data/image_stained/',
                 img_cached=None,
                 transform=None,
                 ):
        self.dataset = sorted(glob.glob("{}*/*/*.npz".format(count_root)))

        if patient is not None:
            self.dataset = [d for d in self.dataset if ((d.split("/")[-2] in patient) or ((d.split("/")[-2], d.split("/")[-1].split("_")[0]) in patient))]

        self.window = window
        self.resolution = resolution

        self.count_root = count_root
        self.img_root = img_root
        self.img_cached = img_cached
        self.transform = transform

        with open(self.count_root + "subtype.pkl", "rb") as f:
            self.subtype = pickle.load(f)

        self.slide = collections.defaultdict(dict)

        for (patient, section) in set([(d.split("/")[-2], d.split("/")[-1].split("_")[0]) for d in self.dataset]):
            self.slide[patient][section] = openslide.open_slide("{}{}/{}/{}_{}.jpg".format(self.img_root, self.subtype[patient], patient, patient, section))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        npz = np.load(self.dataset[index])
        count = npz["count"]
        pixel = npz["pixel"]
        patient = npz["patient"][0]
        section = npz["section"][0]
        coord = npz["index"]

        slide = self.slide[patient][section]
        X = slide.read_region((pixel[0] - self.window  // 2, pixel[1] - self.window  // 2), 0, (self.window , self.window ))
        X = X.convert("RGB")

        he = X

        if self.resolution != 224:
            he = torchvision.transforms.Resize((self.resolution, self.resolution))(he)
        cached_image = "{}{}/{}/{}/{}/{}_{}_{}.jpg".format(self.img_cached, self.subtype[patient], patient, self.window, self.resolution, section, coord[0], coord[1])
        pathlib.Path(cached_image.strip(cached_image.split('/')[-1])).mkdir(parents=True, exist_ok=True)

        he.save(cached_image)

        return X, count

class Spatial(torch.utils.data.Dataset):
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
        self.dataset = sorted(glob.glob("{}*/*/*.npz".format(count_root)))

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

        with open("data/count_filtered/subtype.pkl", "rb") as f:
            self.subtype = pickle.load(f)

        with open("data/count_filtered/gene.pkl", "rb") as f:
            self.ensg_names = pickle.load(f)

        self.mean_expression = np.load('data/count_filtered/mean_expression.npy')

        self.gene_names = list(map(lambda x: ensembl.symbol[x], self.ensg_names))

        keep_gene = set(list(zip(*sorted(zip(self.mean_expression, range(self.mean_expression.shape[0])))[::-1][:self.gene_filter]))[1])

        self.keep_bool = np.array([i in keep_gene for i in range(len(self.gene_names))])

        self.ensg_keep = [n for (n, f) in zip(self.ensg_names, self.keep_bool) if f]
        self.gene_keep = [n for (n, f) in zip(self.gene_names, self.keep_bool) if f]

        if self.aux_ratio != 0:
            self.aux_nums = int((len(self.gene_names) - self.gene_filter) * self.aux_ratio)
            aux_gene = set(list(zip(*sorted(zip(self.mean_expression, range(self.mean_expression.shape[0])))[::-1][self.gene_filter:self.gene_filter + self.aux_nums]))[1])
            self.aux_bool = np.array([i in aux_gene for i in range(len(self.gene_names))])
            self.ensg_aux = [n for (n, f) in zip(self.ensg_names, self.aux_bool) if f]
            self.gene_aux = [n for (n, f) in zip(self.gene_names, self.aux_bool) if f]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        npz = np.load(self.dataset[index])
        count = npz["count"]
        pixel = npz["pixel"]
        patient = npz["patient"][0]
        section = npz["section"][0]
        coord = npz["index"]

        if self.resolution != 224:
            cached_image = "{}{}/{}/{}/{}/{}_{}_{}.jpg".format(self.img_root, self.subtype[patient], patient, self.window, self.resolution, section, coord[0], coord[1])
            X = PIL.Image.open(cached_image)
        else:
            cached_image = "{}{}/{}/{}/{}_{}_{}.jpg".format(self.img_root, self.subtype[patient], patient, self.window, section, coord[0], coord[1])
            X = PIL.Image.open(cached_image)

        if self.transform is not None:
            X = self.transform(X)

        if X.shape[1] != 224:
            X = torchvision.transforms.Resize((224, 224))(X)

        coord = torch.as_tensor(coord)
        index = torch.as_tensor([index])

        keep_count = count[self.keep_bool]
        y = torch.as_tensor(keep_count, dtype=torch.float)
        y = torch.log(1 + y)

        if self.normalization is not None:
            y = (y - self.normalization[0]) / self.normalization[1]

        if self.aux_ratio != 0:
            aux_count = count[self.aux_bool]
            aux = torch.as_tensor(aux_count, dtype=torch.float)
            aux = torch.log(1 + aux)

            return X, y, aux, coord, index, patient, section, pixel
        else:
            return X, y, coord, index, patient, section, pixel


import torchvision
import torch
import utils
import torch.nn.functional as F
import efficientnet_pytorch
import pytorch_pretrained_vit

def set_models(name, pretrained):
    if name == 'efficientnet_b4':
        model = efficientnet_pytorch.EfficientNet.from_name('efficientnet-b4') # no pretrained

        if pretrained:
            model = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b4')

    elif name == "inception_v3":
        model = torchvision.models.__dict__[name](pretrained=False, aux_logits=False)

        if pretrained:
            model = torchvision.models.__dict__[name](pretrained=True, aux_logits=False)

    elif name == "ViT":
        model = pytorch_pretrained_vit.ViT('B_16', pretrained=False)

        if pretrained:
            model = pytorch_pretrained_vit.ViT('B_16', pretrained=True)

    else:
        model = torchvision.models.__dict__[name](pretrained=False)

        if pretrained:
            model = torchvision.models.__dict__[name](pretrained=True)

    return model


class AuxNet(torch.nn.Module):
    def __init__(self, input_dim, output, aux_output):
        super(AuxNet, self).__init__()
        self.fc = torch.nn.Linear(input_dim, output)
        self.aux_fc = torch.nn.Linear(input_dim, aux_output)

    def forward(self, x):
        y = self.fc(x)
        aux = self.aux_fc(x)
        return y, aux


def set_out_features(model, outputs):
    if (isinstance(model, torchvision.models.AlexNet) or
            isinstance(model, torchvision.models.VGG)):
        inputs = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(inputs, outputs, bias=True)
        return model

    elif (isinstance(model, torchvision.models.ResNet) or
          isinstance(model, torchvision.models.Inception3)):
        inputs = model.fc.in_features
        model.fc = torch.nn.Linear(inputs, outputs, bias=True)
        return model

    elif isinstance(model, torchvision.models.DenseNet):
        inputs = model.classifier.in_features
        model.classifier = torch.nn.Linear(inputs, outputs, bias=True)
        return model

    elif isinstance(model, efficientnet_pytorch.EfficientNet):
        inputs = model._fc.in_features
        model._fc = torch.nn.Linear(in_features=inputs, out_features=outputs, bias=True)
        return model

    elif isinstance(model, pytorch_pretrained_vit.ViT):
        inputs = model.fc.in_features
        model.fc = torch.nn.Linear(in_features=inputs, out_features=outputs, bias=True)
        return model


class MyDenseNetConv(torch.nn.Module):
    def __init__(self, fixed_extractor=True):
        super(MyDenseNetConv, self).__init__()
        original_model = torchvision.models.densenet121(pretrained=True)
        self.features = torch.nn.Sequential(*list(original_model.children())[:-1])

        if fixed_extractor:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7).view(x.size(0), -1)
        return x


def setup(train_patients, test_patients, args, device, cv=False):
    train_dataset = Spatial(train_patients,
                            count_root='training/counts/',
                            img_root='training/images/',
                            window=args.window,
                            resolution=args.resolution,
                            gene_filter=args.gene_filter,
                            aux_ratio=args.aux_ratio,
                            transform=torchvision.transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch,
                                               num_workers=args.workers,
                                               shuffle=True)

    mean, std, count_mean, count_std = get_mean_and_std(train_loader, args)

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomApply([torchvision.transforms.RandomRotation((90, 90))]),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std)])

    train_dataset = Spatial(train_patients,
                            count_root='training/counts/',
                            img_root='training/images/',
                            window=args.window,
                            resolution=args.resolution,
                            gene_filter=args.gene_filter,
                            aux_ratio=args.aux_ratio,
                            transform=train_transform,
                            normalization=[count_mean, count_std])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch,
                                               num_workers=args.workers,
                                               shuffle=True)

    val_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std)])

    if cv:
        test_dataset = Spatial(test_patients,
                               count_root='training/counts/',
                               img_root='training/images/',
                               window=args.window,
                               resolution=args.resolution,
                               gene_filter=args.gene_filter,
                               aux_ratio=args.aux_ratio,
                               transform=val_transform,
                               normalization=[count_mean, count_std])
    else:
        test_dataset = Spatial(test_patients,
                               count_root='test/counts/',
                               img_root='test/images/',
                               window=args.window,
                               resolution=args.resolution,
                               gene_filter=args.gene_filter,
                               aux_ratio=args.aux_ratio,
                               transform=val_transform,
                               normalization=[count_mean, count_std])

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch,
                                              num_workers=args.workers,
                                              shuffle=True)

    architecture = set_models(args.model, args.pretrained)
    model = set_out_features(architecture, args.gene_filter)

    if args.model == 'efficientnet_b4':
        if args.finetuning == 'ftfc':
            for param in model.parameters():
                param.requires_grad = False
        
            for cnt, child in enumerate(model.children()):
                if cnt >= 7:
                    for param in child.parameters():
                        param.requires_grad = True

        elif args.finetuning == 'ftconv':
            for param in model.parameters():
                param.requires_grad = False

            for param in model._blocks[15:].parameters():
                param.requires_grad = True

            for cnt, child in enumerate(model.children()):
                if cnt >= 3:
                    for param in child.parameters():
                        param.requires_grad = True

        elif args.finetuning == 'ftall':
            for param in model.parameters():
                param.requires_grad = True

    if args.aux_ratio != 0:
        model._fc = AuxNet(model._fc.in_features, train_dataset.gene_filter, train_dataset.aux_nums)

    model = torch.nn.DataParallel(model)
    model.to(device)

    criterion = torch.nn.MSELoss()
    optim = torch.optim.__dict__['SGD'](model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-6)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=5)

    return model, train_loader, test_loader, optim, lr_scheduler, criterion


import glob
import collections

def get_cv_resluts(patients, cv_folds, args, device):
    fold = [patients[f::cv_folds] for f in range(cv_folds)]

    total_loss = []
    total_aMAE = []
    total_aRMSE = []
    total_aCC = []

    print("### START CROSS VALIDATION:")
    print()

    for f in range(cv_folds):
        print("Fold ##{}".format(f))
        print('=' * 10)
        train = [fold[i] for i in range(cv_folds) if i != f]
        train = [i for j in train for i in j]
        test = fold[f]
        print("Train patients: ", train)
        print("Test patients: ", test)
        print("Parameters: ", args)
        test_loss, test_aMAE, test_aRMSE, test_aCC = train_cv_folds(train, test, args, device)

        total_loss.append(test_loss)
        total_aMAE.append(test_aMAE)
        total_aRMSE.append(test_aRMSE)
        total_aCC.append(test_aCC)

        print()

    print(total_loss)
    print()
    print(total_aMAE)
    print()
    print(total_aRMSE)
    print()
    print(total_aCC)
    print()

    best_loss_epoch = np.argmin(np.vstack(total_loss).mean(0))
    best_aMAE_epoch = np.argmin(np.vstack(total_aMAE).mean(0))
    best_aRMSE_epoch = np.argmin(np.vstack(total_aRMSE).mean(0))
    best_aCC_epoch = np.argmax(np.vstack(total_aCC).mean(0))

    return best_loss_epoch, best_aMAE_epoch, best_aRMSE_epoch, best_aCC_epoch


def train_cv_folds(train_patients, test_patients, args, device):
    total_loss = []
    total_aMAE = []
    total_aRMSE = []
    total_aCC = []

    model, train_loader, test_loader, optim, lr_scheduler, criterion = setup(train_patients, test_patients, args, device, cv=True)

    for epoch in range(args.cv_epochs):
        if args.debug and epoch == 3:
            break
        print("Epoch #" + str(epoch + 1) + ":")
        train_loss, train_aMAE, train_aRMSE, train_aCC = fit(model, train_loader, optim, criterion, args, device)
        lr_scheduler.step()
        test_loss, test_aMAE, test_aRMSE, test_aCC = validate(model, test_loader, criterion, args, device)
        total_loss.append(test_loss)
        total_aMAE.append(test_aMAE)
        total_aRMSE.append(test_aRMSE)
        total_aCC.append(test_aCC)

        torch.cuda.empty_cache()

    return total_loss, total_aMAE, total_aRMSE, total_aCC


def get_spatial_patients():
    """
    Returns a dict of patients to sections.
    The keys of the dict are patient names (str), and the values are lists of
    section names (str).
    """
    patient_section = map(lambda x: x.split("/")[-1].split(".")[0].split("_"), glob.glob("data/image_raw/*/*/*_*.jpg"))
    patient = collections.defaultdict(list)
    for (p, s) in patient_section:
        patient[p].append(s)
    return patient


def patient_or_section(name):
    if "_" in name:
        return tuple(name.split("_"))
    return name


def get_sections(patients, testpatients):
    train_patients = []
    test_patients = []
    for (i, p) in enumerate(patients):
        for s in patients[p]:
            if p in testpatients:
                test_patients.append((p, s))
            else:
                train_patients.append((p, s))

    print('Train patients: ', train_patients)
    print('Test patients: ', test_patients)

    return train_patients, test_patients


def cv_split(patients, cv_folds):
    fold = [patients[f::cv_folds] for f in range(cv_folds)]
    for f in range(cv_folds):
        print("Fold #{}".format(f))
        train = [fold[i] for i in range(5) if i != f]
        train = [i for j in train for i in j]
        test = fold[f]
    return train, test


class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """

    def __init__(
            self, optimizer, patience=5, min_lr=1e-6, factor=0.1
    ):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
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

    def __call__(self, val_loss):  # only if there is one better than others in patience epoch, stop, model.checkpoint can save the best
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


# main program
def run_spatial(args=None):
    ### Seed ###
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ### Select device for computation ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### Split patients into train/test ###
    patients = sorted(get_spatial_patients().keys())
    test_patients = ["BC23450", "BC23903"]
    train_patients = [p for p in patients if p not in test_patients]

    print("Train patients: ", train_patients)
    print("Test patients: ", test_patients)
    print("Parameters: ", args)
    print()

    # ###   cross-validation to get the best epoch

    best_loss_epoch, best_aMAE_epoch, best_aRMSE_epoch, best_aCC_epoch = get_cv_resluts(
        train_patients, args.cv_fold, args, device
    )

    best_epoch = math.ceil(
        np.mean(
            np.array(
                (
                    [best_loss_epoch, best_aMAE_epoch, best_aRMSE_epoch, best_aCC_epoch]
                )
            )
        )
    )
    print("Best CV epoch: ", best_epoch)
    print()

    # best_epoch = 10

    ###     cross-validation to get the best epoch

    print("### START MAIN PROGRAM:")
    print()
    print("Train patients: ", train_patients)
    print("Test patients: ", test_patients)
    print("Parameters: ", args)

    ### main network
    model, train_loader, test_loader, optim, lr_scheduler, criterion = setup(
        train_patients, test_patients, args, device
    )

    if best_epoch <= 3:  # for debug
        best_epoch = 3

    for epoch in range(best_epoch):

        if args.debug and epoch == 3:
            break

        print("Epoch #" + str(epoch + 1) + ":")
        train_loss = fit(model, train_loader, optim, criterion, args, device)
        lr_scheduler.step()
        # here test will save should change the file name
        test_loss = test(model, test_loader, criterion, device, args, best_epoch)

    ### TODO: best_epoch = 0, skip the for loop and direct save model?
    # if args.debug:
    #     pass
    # else:
    torch.save(model, args.pred_root + "/model.pkl")
    print()

    torch.cuda.empty_cache()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process the paths.")

    parser.add_argument("--seed", type=int, default=0, help="seed for reproduction")

    parser.add_argument(
        "--cv_fold",
        type=int,
        default=5,
        help="cv fold for cross-validation",
    )

    parser.add_argument(
        "--batch", type=int, default=32, help="training batch size"
    )

    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="learning rate"
    )

    parser.add_argument(
        "--cv_epochs",
        type=int,
        default=50,
        help="number of cross-validation epochs",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="number of workers for dataloader",
    )

    parser.add_argument(
        "--window",
        type=int,
        default=299,
        help="window size",
    )  # try 128 150 224 299 512 (smaller, normal, and bigger)

    parser.add_argument(
        "--resolution",
        type=int,
        default=224,
        help="resolution",
    )  # try

    parser.add_argument(
        "--model",
        type=str,
        default="efficientnet_b4",
        help="choose different model",
    )  # alexnet, vgg16, resnet101, densenet121, inception_v3, efficientnet_b7

    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="use ImageNet pretrained model?",
    )

    parser.add_argument(
        "--finetuning",
        type=str,
        default=None,
        help="use ImageNet pretrained model with fine tuning fcs?",
    )

    parser.add_argument(
        "--gene_filter",
        default=250,
        type=int,
        help="specific predicted main genes (defalt use all the rest for aux tasks)",
    )

    parser.add_argument(
        "--aux_ratio",
        default=1,
        type=float,
        help="specific the number of aux genes",
    )

    parser.add_argument(
        "--aux_weight",
        default=1,
        type=float,
        help="specific the loss weight of aux genes",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="number of epochs",
    )

    parser.add_argument(
        "--pred_root",
        type=str,
        default="/srv/scratch/bic/mmr/hist2t/",
        help="root for prediction outputs",
    )

    parser.add_argument("--debug", action="store_true", help="debug")
    parser.add_argument("-f")
    args = parser.parse_args()

    # set different log name

    pathlib.Path(os.path.dirname(args.pred_root)).mkdir(parents=True, exist_ok=True)

    run_spatial(args)
