import os
import gc
import numpy as np
from openmm.unit import *
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader

from utils import ani1x_iter_data_buckets, anidataloader

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') 

ATOM_DICT = {1: 'H', 6: 'C', 7: 'N', 8: 'O'}
PAIR_DICT = {('Br', -1): 0, ('Br', 0): 1, ('C', -1): 2, ('C', 0): 3, ('C', 1): 4, ('Ca', 2): 5, ('Cl', -1): 6,
            ('Cl', 0): 7, ('F', -1): 8, ('F', 0): 9, ('H', 0): 10, ('I', -1): 11, ('I', 0): 12, ('K', 1): 13,
            ('Li', 1): 14, ('Mg', 2): 15, ('N', -1): 16, ('N', 0): 17, ('N', 1): 18, ('Na', 1): 19, ('O', -1): 20,
            ('O', 0): 21, ('O', 1): 22, ('P', 0): 23, ('P', 1): 24, ('S', -1): 25, ('S', 0): 26, ('S', 1): 27}


class PretrainData(Dataset):
    def __init__(self, species, positions, smiles, std):
        self.species = species
        self.positions = positions
        self.smiles = smiles
        self.std = std
    
    def __getitem__(self, index):
        original_positions = self.positions[index]
        atoms = self.species[index]

        # add noise
        noise = np.random.normal(0, self.std, original_positions.shape)
        pos = original_positions + noise

        x = torch.tensor(atoms, dtype=torch.long)
        pos = torch.tensor(pos, dtype=torch.float)
        noise = torch.tensor(noise, dtype=torch.float)
        
        data = Data(x=x, pos=pos, noise=noise)
        return data

    def __len__(self):
        return len(self.positions)


class PretrainDataWrapper:
    def __init__(self, batch_size, num_workers, valid_size, ani1, ani1x, std, seed, **kwargs):
        super(PretrainDataWrapper, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.ani1 = ani1
        self.ani1x = ani1x
        self.std = std
        self.seed = seed

    def get_data_loaders(self):
        random_state = np.random.RandomState(seed=self.seed)

        n_mol = 0
        train_species, valid_species = [], []
        train_positions, valid_positions = [], []
        train_smiles, valid_smiles = [], []
        
        # read ANI-1 data
        if self.ani1:
            print('Loading ANI-1 data...')
            hdf5_files = [f for f in os.listdir('../ANI-1_release') if f.endswith('.h5')]
            for file_name in hdf5_files:
                print('reading:', file_name)
                h5_loader = anidataloader(os.path.join('../ANI-1_release', file_name))
                for data in h5_loader:
                    n_mol += 1
                    if n_mol % 1000 == 0:
                        print('Loading # molecule %d' % n_mol)
                    
                    X = data['coordinates']
                    S = data['species']
                    E = data['energies']

                    n_conf = E.shape[0]
                    indices = list(range(n_conf))
                    random_state.shuffle(indices)
                    split = int(np.floor(self.valid_size * n_conf))
                    valid_idx, train_idx = indices[:split], indices[split:]

                    species = [PAIR_DICT[(ele, 0)] for ele in S]
                    train_species.extend([species] * len(train_idx))
                    train_smiles.extend([''] * len(train_idx))
                    for i in train_idx:
                        train_positions.append(X[i])

                    species = [PAIR_DICT[(ele, 0)] for ele in S]
                    valid_species.extend([species] * len(valid_idx))
                    valid_smiles.extend([''] * len(valid_idx))
                    for i in valid_idx:
                        valid_positions.append(X[i])
                
                h5_loader.cleanup()

        # read ANI-1x data
        if self.ani1x:
            print('Loading ANI-1x data...')
            data_path = '../ANI-1x/ani1x_release.h5'
            data_keys = ['wb97x_dz.energy', 'wb97x_dz.forces']

            # extracting DFT/DZ energies and forces
            for data in ani1x_iter_data_buckets(data_path, keys=data_keys):
                n_mol += 1
                if n_mol % 1000 == 0:
                    print('Loading # molecule %d' % n_mol)
                
                X = data['coordinates']
                S = data['atomic_numbers']
                E = data['wb97x_dz.energy']
                S = [PAIR_DICT[(ATOM_DICT[c], 0)] for c in S]

                n_conf = E.shape[0]
                indices = list(range(n_conf))
                random_state.shuffle(indices)
                split = int(np.floor(self.valid_size * n_conf))
                valid_idx, train_idx = indices[:split], indices[split:]

                train_species.extend([S] * len(train_idx))
                train_smiles.extend([''] * len(train_idx))
                for i in train_idx:
                    train_positions.append(X[i])
                
                valid_species.extend([S] * len(valid_idx))
                valid_smiles.extend([''] * len(valid_idx))
                for i in valid_idx:
                    valid_positions.append(X[i])

        print("# molecules:", n_mol)
        print("# train conformations:", len(train_species))
        print("# valid conformations:", len(valid_species))

        train_dataset = PretrainData(
            species=train_species, positions=train_positions, 
            smiles=train_smiles, std=self.std
        )
        valid_dataset = PretrainData(
            species=valid_species, positions=valid_positions, 
            smiles=valid_smiles, std=self.std
        )

        train_loader = PyGDataLoader(
            train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, 
            shuffle=True, drop_last=True, pin_memory=True, persistent_workers=(self.num_workers > 0)
        )
        valid_loader = PyGDataLoader(
            valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers, 
            shuffle=False, drop_last=False, pin_memory=True, persistent_workers=(self
