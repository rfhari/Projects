
import torch
from torch import optim
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from featurizer import *
import pandas as pd
from torch.autograd import Variable
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset, TensorDataset
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
from rdkit import DataStructs
from torchvision import transforms, datasets

torch.manual_seed(0)  
np.random.seed(0)

def load_all():
    f = pd.read_csv("delaney-processed modified.csv")
    all_smiles = np.array(f['smiles'])
    all_lab = np.array(f['labels'])
    valid_mols = [Chem.MolFromSmiles(sm) for sm in all_smiles if Chem.MolFromSmiles(sm) is not None]
    valid_labels = [all_lab[i] for i, sm in enumerate(all_smiles) if Chem.MolFromSmiles(sm) is not None]
    return valid_mols, valid_labels

def morgan_featurization(mol, flag=0, cutoff=3, features=1024):
    fps = []
    fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, cutoff, nBits = features))
    matrices = []
    for i in range(len(fps)):
        array = np.zeros((1,));
        DataStructs.ConvertToNumpyArray(fps[i], array);
        matrices.append(array)
    matrices = np.asarray(matrices).reshape(32, 32)
    return matrices 

def morgan_featurizer(mols):
    mol_feat, lab = [], []
    for i, mol in enumerate(mols):
        feat2 = morgan_featurization(mol)
        mol_feat.append(feat2.reshape(1024*1))
    mol_feat = np.array(mol_feat)
    return mol_feat

def frag_ecfp(mol):
    res = list(BRICSDecompose(mol, returnMols=True))
    mol_feat = morgan_featurizer(res)
    frag_ecfp_feat = np.sum(mol_feat, axis=0)
    return frag_ecfp_feat

def frag_mol(mol):
    res = list(BRICSDecompose(mol, returnMols = False, singlePass = True))
    u = set(res)
    return u, len(u)

x_train0, y_train0 = load_all()
frag_ecfp_feat = []
for (x, y)   in zip (x_train0, y_train0):
    new_frag = frag_ecfp(x)
    frag_ecfp_feat.append(new_frag)