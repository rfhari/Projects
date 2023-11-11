
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from rdkit.Chem import MACCSkeys
from featurizer import * 
import matplotlib.pyplot as plt
from rdkit import DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
from rdkit.Chem import MACCSkeys

def load_all():
    f = pd.read_csv("delaney-processed modified.csv")
    all_smiles = np.array(f['smiles'])
    all_lab = np.array(f['labels'])
    valid_mols = [Chem.MolFromSmiles(sm) for sm in all_smiles if Chem.MolFromSmiles(sm)]
    valid_labels = [all_lab[i] for i, sm in enumerate(all_smiles) if Chem.MolFromSmiles(sm)]
    return valid_mols, valid_labels

def morgan_featurizer(mols, labels):
    mol_feat, lab = [], []
    for i, mol in enumerate(mols):
        feat2, _ = morgan_featurization(mol, labels[i])
        mol_feat.append(feat2.reshape(1024*2))
        lab.append(labels[i])
    mol_feat = np.array(mol_feat)
    print(np.shape(mol_feat))
    lab = np.array(lab).reshape(-1, 1)
    return mol_feat, lab

mols, labels = load_all()
# mol_feat, label_feat = morgan_featurizer(mols, labels)
mol_feat = [MACCSkeys.GenMACCSKeys(m) for m in mols]
radius, clusters = 5, 10
# fig1, ax1 = plt.subplots()
kmeans = KMeans(n_clusters = clusters, random_state=0).fit(mol_feat)
y_pred = kmeans.predict(mol_feat)
k_mean_labels = kmeans.labels_

# -----------------------------------------

# c0 = [mols[i] for i, k in enumerate(kmeans.labels_) if k==0]
# fps = [GetMorganFingerprint(x, radius) for x in c0]

# def distij(i,j,fps=fps):
#     return 1-DataStructs.TanimotoSimilarity(fps[i],fps[j])

# tani0=[]
# for i in range(len(c0)):
#     for j in range(len(c0)):
#         tani0.append(distij(i, j))

# ----------------------------------------- 

# c1 = [mols[i] for i, k in enumerate(kmeans.labels_) if k==1]
# fps = [GetMorganFingerprint(x, radius) for x in c1]

# def distij(i,j,fps=fps):
#     return 1-DataStructs.TanimotoSimilarity(fps[i],fps[j])

# tani1=[]
# for i in range(len(c1)):
#     for j in range(len(c1)):
#         tani1.append(distij(i, j))
        
# # -----------------------------------------
       
# c2 = [mols[i] for i, k in enumerate(kmeans.labels_) if k==2]
# fps = [GetMorganFingerprint(x, radius) for x in c2]

# def distij(i,j,fps=fps):
#     return 1-DataStructs.TanimotoSimilarity(fps[i],fps[j])

# tani2=[]
# for i in range(len(c2)):
#     for j in range(len(c2)):
#         tani2.append(distij(i, j))

# -----------------------------------------

# c3 = [mols[i] for i, k in enumerate(kmeans.labels_) if k==3]
# fps = [GetMorganFingerprint(x, radius) for x in c3]

# def distij(i,j,fps=fps):
#     return 1-DataStructs.TanimotoSimilarity(fps[i],fps[j])

# tani3=[]
# for i in range(len(c3)):
#     for j in range(len(c3)):
#         tani3.append(distij(i, j))

# # -----------------------------------------
# c4 = [mols[i] for i, k in enumerate(kmeans.labels_) if k==4]
# fps = [GetMorganFingerprint(x, radius) for x in c4]

# def distij(i,j,fps=fps):
#     return 1-DataStructs.TanimotoSimilarity(fps[i],fps[j])

# tani4=[]
# for i in range(len(c4)):
#     for j in range(len(c4)):
#         tani4.append(distij(i, j))
# -----------------------------------------
# tani0 = np.asarray(tani0).reshape(-1, len(c0))    
# tani1 = np.asarray(tani1).reshape(-1, len(c1))
# tani2 = np.asarray(tani2).reshape(-1, len(c2))
# # tani3 = np.asarray(tani3).reshape(-1, len(c3))
# # tani4 = np.asarray(tani4).reshape(-1, len(c4))

# t0_mean = np.mean(tani0, axis=1)
# t1_mean = np.mean(tani1, axis=1)
# t2_mean = np.mean(tani2, axis=1)
# t3_mean = np.mean(tani3, axis=1)
# t4_mean = np.mean(tani4, axis=1)

# Total_tani = []
# for cluster_num in range(clusters):
#     c = [mols[i] for i, k in enumerate(kmeans.labels_) if k==cluster_num]
#     # fps = [MACCSkeys.GenMACCSKeys(x) for x in c]
#     fps = [GetMorganFingerprint(x, 5) for x in c]
#     def distij (i, j, fps=fps):
#         return DataStructs.TanimotoSimilarity (fps[i], fps[j]) #Note for me - this is Tanimoto Coeff ie higher it is better it is
#     tani = []
#     for i in range(len(c)):
#         for j in range(len(c)):
#             tani.append(distij(i, j))
#     tani = np.asarray(tani).reshape(-1, len(c))
#     tani_mean = np.mean(tani)
#     Total_tani.append(tani_mean)