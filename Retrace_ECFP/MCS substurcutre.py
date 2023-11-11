#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rdkit import Chem
import numpy as np
from rdkit.Chem import Draw, AllChem, MCS
import pandas as pd
import matplotlib.pyplot as plt
import random
from rdkit.Chem import Draw, AllChem

def load_all():
    f = pd.read_csv("data_gap.csv")
    sm = np.array(f['smiles'])
    lab = np.array(f['label'])
    return sm, lab

sm, lab = load_all()
mols = [Chem.MolFromSmiles(mol) for mol in sm if Chem.MolFromSmiles(mol) is not None]
p = MCS.FindMCS(mols, minNumAtoms=2, maximize='atoms', atomCompare='elements', bondCompare='bondtypes', matchValences=True,
        ringMatchesRingOnly=True, completeRingsOnly=False, threshold=0.85)