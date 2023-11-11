import numpy as np
import pandas as pd
import functools
import torch
import pickle
from torch.utils.data import Dataset
from torch_geometric.data import Dataset as torch_Dataset
from torch_geometric.data import Data, DataLoader as torch_DataLoader
import sys, json, os
from pymatgen.core.structure import Structure
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering as SPCL
import warnings
from Mat2Spec.utils import *
from os import path

from Mat2Spec.augmentation import RotationTransformation, PerturbStructureTransformation, RemoveSitesTransformation,TranslateSitesTransformation

# Note: this file for data loading is modified from https://github.com/superlouis/GATGNN/blob/master/gatgnn/data.py

# gpu_id = 0
# device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
device = set_device()

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

class ELEM_Encoder:
    def __init__(self):
        self.elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl',
                         'Ar', 'K',
                         'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
                         'Kr', 'Rb',
                         'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I',
                         'Xe', 'Cs',
                         'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                         'Hf', 'Ta',
                         'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac',
                         'Th', 'Pa',
                         'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']  # 103
        self.e_arr = np.array(self.elements)

    def encode(self, composition_dict):  # from formula to composition, which is a vector of length 103
        answer = [0] * len(self.elements)

        elements = [str(i) for i in composition_dict.keys()]
        counts = [j for j in composition_dict.values()]
        total = sum(counts)

        for idx in range(len(elements)):
            elem = elements[idx]
            ratio = counts[idx] / total
            idx_e = self.elements.index(elem)
            answer[idx_e] = ratio
        return torch.tensor(answer).float().view(1, -1)

    def decode_pymatgen_num(self, tensor_idx):  # from ele_num to ele_name
        idx = (tensor_idx - 1).cpu().tolist()
        return self.e_arr[idx]


class DATA_normalizer:
    def __init__(self, array):
        tensor = torch.tensor(array)
        self.mean = torch.mean(tensor, dim=0).float()
        self.std = torch.std(tensor, dim=0).float()

    def reg(self, x):
        return x.float()

    def log10(self, x):
        return torch.log10(x)

    def delog10(self, x):
        return 10 * x

    def norm(self, x):
        return (x - self.mean) / self.std

    def denorm(self, x):
        return x * self.std + self.mean


class METRICS:
    def __init__(self, c_property, epoch, torch_criterion, torch_func, device):
        self.c_property = c_property
        self.criterion = torch_criterion
        self.eval_func = torch_func
        self.dv = device
        self.training_measure1 = torch.tensor(0.0).to(device)
        self.training_measure2 = torch.tensor(0.0).to(device)
        self.valid_measure1 = torch.tensor(0.0).to(device)
        self.valid_measure2 = torch.tensor(0.0).to(device)

        self.training_counter = 0
        self.valid_counter = 0

        self.training_loss1 = []
        self.training_loss2 = []
        self.valid_loss1 = []
        self.valid_loss2 = []
        self.duration = []
        self.dataframe = self.to_frame()

    def __str__(self):
        x = self.to_frame()
        return x.to_string()

    def to_frame(self):
        metrics_df = pd.DataFrame(list(zip(self.training_loss1, self.training_loss2,
                                           self.valid_loss1, self.valid_loss2, self.duration)),
                                  columns=['training_1', 'training_2', 'valid_1', 'valid_2', 'time'])
        return metrics_df

    def set_label(self, which_phase, graph_data):
        use_label = graph_data.y
        return use_label

    def save_time(self, e_duration):
        self.duration.append(e_duration)

    def __call__(self, which_phase, tensor_pred, tensor_true, measure=1):
        if measure == 1:
            if which_phase == 'training':
                loss = self.criterion(tensor_pred, tensor_true)
                self.training_measure1 += loss
            elif which_phase == 'validation':
                loss = self.criterion(tensor_pred, tensor_true)
                self.valid_measure1 += loss
        else:
            if which_phase == 'training':
                loss = self.eval_func(tensor_pred, tensor_true)
                self.training_measure2 += loss
            elif which_phase == 'validation':
                loss = self.eval_func(tensor_pred, tensor_true)
                self.valid_measure2 += loss
        return loss

    def reset_parameters(self, which_phase, epoch):
        if which_phase == 'training':
            # AVERAGES
            t1 = self.training_measure1 / (self.training_counter)
            t2 = self.training_measure2 / (self.training_counter)

            self.training_loss1.append(t1.item())
            self.training_loss2.append(t2.item())
            self.training_measure1 = torch.tensor(0.0).to(self.dv)
            self.training_measure2 = torch.tensor(0.0).to(self.dv)
            self.training_counter = 0
        else:
            # AVERAGES
            v1 = self.valid_measure1 / (self.valid_counter)
            v2 = self.valid_measure2 / (self.valid_counter)

            self.valid_loss1.append(v1.item())
            self.valid_loss2.append(v2.item())
            self.valid_measure1 = torch.tensor(0.0).to(self.dv)
            self.valid_measure2 = torch.tensor(0.0).to(self.dv)
            self.valid_counter = 0

    def save_info(self):
        with open('MODELS/metrics_.pickle', 'wb') as metrics_file:
            pickle.dump(self, metrics_file)


class GaussianDistance(object):
    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)  # int((dmax-dmin) / step) + 1
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        # print(distances.shape) [nbr, nbr]
        # x = distances[..., np.newaxis] [nbr, nbr, 1]
        # print(self.filter.shape)
        # print((x-self.filter).shape)
        return np.exp(-(distances[..., np.newaxis] - self.filter) ** 2 / self.var ** 2)


class AtomInitializer(object):
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())  # 100
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class CIF_Lister(Dataset):
    def __init__(self, crystals_ids, full_dataset, df=None):
        self.crystals_ids = crystals_ids
        self.full_dataset = full_dataset
        self.material_ids = df.iloc[crystals_ids].values[:, 0].squeeze()  # MP-xxx

    def __len__(self):
        return len(self.crystals_ids)

    def extract_ids(self, original_dataset):
        names = original_dataset.iloc[self.crystals_ids]
        return names

    def __getitem__(self, idx):
        i = self.crystals_ids[idx]
        material1, material2 = self.full_dataset[i]

        n_features = material1[0][0]
        e_features = material1[0][1]  # [n_atom, nbr, 41]
        e_features = e_features.view(-1, 41)
        a_matrix = material1[0][2]

        groups = material1[1]
        enc_compo = material1[2]  # normalize feat
        coordinates = material1[3]
        y = material1[4]  # target

        graph_crystal1 = Data(x=n_features, y=y, edge_attr=e_features, edge_index=a_matrix, global_feature=enc_compo, \
                             cluster=groups, num_atoms=torch.tensor([len(n_features)]).float(), coords=coordinates,
                             the_idx=torch.tensor([float(i)]))
        
        n_features = material2[0][0]
        e_features = material2[0][1]  # [n_atom, nbr, 41]
        e_features = e_features.view(-1, 41)
        a_matrix = material2[0][2]

        groups = material2[1]
        enc_compo = material2[2]  # normalize feat
        coordinates = material2[3]
        y = material2[4]  # target

        graph_crystal2 = Data(x=n_features, y=y, edge_attr=e_features, edge_index=a_matrix, global_feature=enc_compo, \
                             cluster=groups, num_atoms=torch.tensor([len(n_features)]).float(), coords=coordinates,
                             the_idx=torch.tensor([float(i)]))

        return graph_crystal1, graph_crystal2

class CIF_Dataset(Dataset):
    def __init__(self, args, pd_data=None, np_data=None, norm_obj=None, normalization=None, max_num_nbr=12, radius=8,
                 dmin=0, step=0.2, cls_num=3, root_dir='DATA/'):
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.pd_data = pd_data
        self.np_data = np_data
        self.ari = AtomCustomJSONInitializer(self.root_dir + 'atom_init.json')
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        self.clusterizer = SPCL(n_clusters=cls_num, random_state=None, assign_labels='discretize')
        self.clusterizer2 = KMeans(n_clusters=cls_num, random_state=None)
        self.encoder_elem = ELEM_Encoder()
        self.update_root = None
        self.args = args
        if self.args.data_src == 'ph_dos_51':
            #self.structures = torch.load('DATA/20210612_ph_dos_51/ph_structures.pt')
            pkl_file = open('../Mat2Spec_DATA/phdos/ph_structures.pkl', 'rb')
            self.structures = pickle.load(pkl_file)
            pkl_file.close()
        
        self.rotater = RotationTransformation()
        self.perturber = PerturbStructureTransformation(distance=1, min_distance=0.0) 
        self.translator = TranslateSitesTransformation(None,None, True)       
        self.masker = RemoveSitesTransformation()

    def __len__(self):
        return len(self.pd_data)

    # @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        cif_id = self.pd_data.iloc[idx][0]
        target = self.np_data[idx]

        catche_data_exist = False

        if self.args.data_src == 'binned_dos_128':
            if path.exists(f'../Mat2Spec_DATA/materials_with_edos_processed/' + cif_id + '.chkpt'):
                catche_data_exist = True
        elif self.args.data_src == 'ph_dos_51':
            if path.exists(f'../Mat2Spec_DATA/materials_with_phdos_processed/' + str(cif_id) + '.chkpt'):
                catche_data_exist = True
        elif self.args.data_src == 'no_label_128':
            if path.exists(f'../Mat2Spec_DATA/materials_without_dos_processed/' + cif_id + '.chkpt'):
                catche_data_exist = True

        if self.args.use_catached_data and catche_data_exist:
            if self.args.data_src == 'binned_dos_128':
                tmp_dist1 = torch.load(f'../Mat2Spec_DATA/materials_with_edos_processed/tmp_dist1_' + cif_id + '.chkpt')
                tmp_dist2 = torch.load(f'../Mat2Spec_DATA/materials_with_edos_processed/tmp_dist2_' + cif_id + '.chkpt')
            elif self.args.data_src == 'ph_dos_51':
                tmp_dist1 = torch.load(f'../Mat2Spec_DATA/materials_with_edos_processed/tmp_dist1_' + cif_id + '.chkpt')
                tmp_dist2 = torch.load(f'../Mat2Spec_DATA/materials_with_edos_processed/tmp_dist2_' + cif_id + '.chkpt')
            elif self.args.data_src == 'no_label_128':
                tmp_dist1 = torch.load(f'../Mat2Spec_DATA/materials_with_edos_processed/tmp_dist1_' + cif_id + '.chkpt')
                tmp_dist2 = torch.load(f'../Mat2Spec_DATA/materials_with_edos_processed/tmp_dist2_' + cif_id + '.chkpt')

            atom_fea_rot_1 = tmp_dist1['atom_fea']
            nbr_fea_rot_1 = tmp_dist1['nbr_fea']
            nbr_fea_idx_rot_1 = tmp_dist1['nbr_fea_idx']
            groups1 = tmp_dist1['groups']
            enc_compo1 = tmp_dist1['enc_compo']
            coordinates1 = tmp_dist1['coordinates']
            target = tmp_dist1['target']
            cif_id = tmp_dist1['cif_id']
            atom_id1 = tmp_dist1['atom_id']

            atom_fea_rot_2 = tmp_dist2['atom_fea']
            nbr_fea_rot_2 = tmp_dist2['nbr_fea']
            nbr_fea_idx_rot_2 = tmp_dist2['nbr_fea_idx']
            groups2 = tmp_dist2['groups']
            enc_compo2 = tmp_dist2['enc_compo']
            coordinates2 = tmp_dist2['coordinates']
            target = tmp_dist2['target']
            cif_id = tmp_dist2['cif_id']
            atom_id2 = tmp_dist2['atom_id']
            
            # return (atom_fea, nbr_fea, nbr_fea_idx), groups, enc_compo, coordinates, target, cif_id, atom_id
            return [(atom_fea_rot_1, nbr_fea_rot_1, nbr_fea_idx_rot_1), groups1, enc_compo1, coordinates1, target, cif_id, atom_id1], [(atom_fea_rot_2, nbr_fea_rot_2, nbr_fea_idx_rot_2), groups2, enc_compo2, coordinates2, target, cif_id, atom_id2]

        if self.args.data_src == 'binned_dos_128':
            with open(os.path.join(self.root_dir + 'materials_with_edos/', 'dos_' + cif_id + '.json')) as json_file:
                data = json.load(json_file)
                crystal = Structure.from_dict(data['structure'])
        
        elif self.args.data_src == 'ph_dos_51':
            print("YES")
            crystal = self.structures[idx]
        
        elif self.args.data_src == 'no_label_128':
            with open(os.path.join(self.root_dir + 'materials_without_dos/', cif_id + '.json')) as json_file:
                data = json.load(json_file)
                crystal = Structure.from_dict(data['structure'])

        # ------------------------------------------------------------------------------------
        # augm changes begin from here
                
        print("crystal test:", crystal[0].specie.number)
        print("crystal:", type(crystal))
        crys = crystal
        crystal = crys.copy()

        # perturbe
        crystal_per_1 =  self.perturber.apply_transformation(crystal)
        crystal_per_2 =  self.perturber.apply_transformation(crystal)

        num_sites = crystal.num_sites

        mask_num = max((1, int(np.floor(0.25*num_sites))))
        indices_trans_1 = np.random.choice(num_sites, mask_num, replace=False)
        indices_trans_2 = np.random.choice(num_sites, mask_num, replace=False)
        
        # translation
        translation_1 = np.random.rand(len(indices_trans_1),3)
        translation_2 = np.random.rand(len(indices_trans_2),3)
        translate_1 = TranslateSitesTransformation(indices_trans_1,translation_1)
        translate_2 = TranslateSitesTransformation(indices_trans_2,translation_2)

        for i in range(3):
            axis = np.zeros(3)
            axis[i] = 1
            rot_ang = np.random.uniform(-90.0, 90.0)
            crystal_rot_1 = translate_1.apply_transformation(self.rotater.apply_transformation(crystal_per_1, axis, rot_ang, angle_in_radians=False))

        for i in range(3):
            axis = np.zeros(3)
            axis[i] = 1
            rot_ang = np.random.uniform(-90.0, 90.0)
            crystal_rot_2 = translate_2.apply_transformation(self.rotater.apply_transformation(crystal_per_2, axis, rot_ang, angle_in_radians=False))

        crystal_rot_1 = crystal
        atom_fea_rot_1 = np.vstack([self.ari.get_atom_fea(crystal_rot_1[i].specie.number)
                              for i in range(len(crystal_rot_1))])

        # atom_fea_rot_1 = torch.Tensor(atom_fea_rot_1)
        all_nbrs_rot_1 = crystal_rot_1.get_all_neighbors(self.radius, include_index=True)
        all_nbrs_rot_1 = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs_rot_1]
        nbr_fea_idx_rot_1, nbr_fea_rot_1 = [], []
        
        for nbr in all_nbrs_rot_1:
            if len(nbr) < self.max_num_nbr:
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(cif_id))
                nbr_fea_idx_rot_1.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea_rot_1.append(list(map(lambda x: x[1], nbr)) +
                               [self.radius + 1.] * (self.max_num_nbr -
                                                     len(nbr)))
            
            else:
                # mask 25% edges                
                nbr_fea_idx_rot_1.append(list(map(lambda x: x[2],
                                            nbr[:self.max_num_nbr])))
                nbr_fea_rot_1.append(list(map(lambda x: x[1],
                                        nbr[:self.max_num_nbr])))

        nbr_fea_idx_rot_1, nbr_fea_rot_1 = np.array(nbr_fea_idx_rot_1), np.array(nbr_fea_rot_1)
        nbr_fea_rot_1 = self.gdf.expand(nbr_fea_rot_1)
        atom_fea_rot_1 = torch.Tensor(atom_fea_rot_1)
        nbr_fea_rot_1 = torch.Tensor(nbr_fea_rot_1)
        # nbr_fea_idx_rot_1 = torch.LongTensor(nbr_fea_idx_rot_1)
        nbr_fea_idx_rot_1 = self.format_adj_matrix(torch.LongTensor(nbr_fea_idx_rot_1))
        
        g_coords1 = crystal_rot_1.cart_coords
        
        # print(g_coords.shape) # [n_atom, 3]
        
        groups1 = [0] * len(g_coords1)
        
        if len(g_coords1) > 2:
            try:
                groups1 = self.clusterizer.fit_predict(g_coords1)
            except:
                groups1 = self.clusterizer2.fit_predict(g_coords1)
        
        groups1 = torch.tensor(groups1).long()  # [n_atom]
        
        # target = torch.Tensor([float(target)])
        target = torch.Tensor(target.astype(float)).view(1, -1)
        coordinates1 = torch.tensor(g_coords1)  # [n_atom, 3]

        atom_fea_rot_2 = np.vstack([self.ari.get_atom_fea(crystal_rot_2[i].specie.number)
                              for i in range(len(crystal_rot_2))])
        
        # mask 25% atoms
    
        atom_fea_rot_2 = torch.Tensor(atom_fea_rot_2)

        all_nbrs_rot_2 = crystal_rot_2.get_all_neighbors(self.radius, include_index=True)
        all_nbrs_rot_2 = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs_rot_2]
        nbr_fea_idx_rot_2, nbr_fea_rot_2 = [], []
        
        for nbr in all_nbrs_rot_2:
            if len(nbr) < self.max_num_nbr:
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(cif_id))
                
                nbr_fea_idx_rot_2.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea_rot_2.append(list(map(lambda x: x[1], nbr)) +
                               [self.radius + 1.] * (self.max_num_nbr -
                                                     len(nbr)))
            else:
                # mask 25% edges
                nbr_fea_idx_rot_2.append(list(map(lambda x: x[2],
                                            nbr[:self.max_num_nbr])))
                nbr_fea_rot_2.append(list(map(lambda x: x[1],
                                        nbr[:self.max_num_nbr])))
        
        nbr_fea_idx_rot_2, nbr_fea_rot_2 = np.array(nbr_fea_idx_rot_2), np.array(nbr_fea_rot_2)
        nbr_fea_rot_2 = self.gdf.expand(nbr_fea_rot_2)
        atom_fea_rot_2 = torch.Tensor(atom_fea_rot_2)
        nbr_fea_rot_2 = torch.Tensor(nbr_fea_rot_2)

        nbr_fea_idx_rot_2 = self.format_adj_matrix(torch.LongTensor(nbr_fea_idx_rot_2))

        g_coords2 = crystal_rot_2.cart_coords
        
        # print(g_coords.shape) # [n_atom, 3]
        
        groups2 = [0] * len(g_coords2)
        
        if len(g_coords2) > 2:
            try:
                groups2 = self.clusterizer.fit_predict(g_coords2)
            except:
                groups2 = self.clusterizer2.fit_predict(g_coords2)
        
        groups2 = torch.tensor(groups2).long()  # [n_atom]
        coordinates2 = torch.tensor(g_coords2)  # [n_atom, 3]


        # return (atom_fea_rot_1, nbr_fea_rot_1, nbr_fea_idx_rot_1), (atom_fea_rot_2, nbr_fea_rot_2, nbr_fea_idx_rot_2), cif_id
               
        # ------------------------------------------------------------------------------------

        #existing work beyond this
        # atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number) for i in range(len(crystal))])

        # atom_fea = torch.Tensor(atom_fea)

        # all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)  # (site, distance, index, image)

        # all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]  # [num_atom in this crystal]
        
        # nbr_fea_idx, nbr_fea = [], []
        
        # for nbr in all_nbrs:
        #     if len(nbr) < self.max_num_nbr:
        #         nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) + [0] * (self.max_num_nbr - len(nbr)))
        #         nbr_fea.append(list(map(lambda x: x[1], nbr)) + [self.radius + 1.] * (self.max_num_nbr - len(nbr)))
        #     else:
        #         nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:self.max_num_nbr])))
        #         nbr_fea.append(list(map(lambda x: x[1], nbr[:self.max_num_nbr])))
        # nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)

        # print(nbr_fea_idx.shape) # [n_atom, nbr]
        # print(nbr_fea.shape) # [n_atom, nbr]
        # nbr_fea = self.gdf.expand(nbr_fea)
        # print(nbr_fea.shape) # [n_atom, nbr, 41]

        # g_coords = crystal.cart_coords
        # print(g_coords.shape) # [n_atom, 3]
        # groups = [0] * len(g_coords)
        # if len(g_coords) > 2:
            # try:
                # groups = self.clusterizer.fit_predict(g_coords)
            # except:
                # groups = self.clusterizer2.fit_predict(g_coords)
        # groups = torch.tensor(groups).long()  # [n_atom]

        # atom_fea = torch.Tensor(atom_fea)
        # nbr_fea = torch.Tensor(nbr_fea)
        # nbr_fea_idx = self.format_adj_matrix(torch.LongTensor(nbr_fea_idx))  # [2, E]

        # target = torch.Tensor(target.astype(float)).view(1, -1)
        # coordinates = torch.tensor(g_coords)  # [n_atom, 3]

        # enc_compo = self.encoder_elem.encode(crystal.composition) # [1, 103]

        enc_compo1 = self.encoder_elem.encode(crystal_rot_1.composition)  # [1, 103]
        enc_compo2 = self.encoder_elem.encode(crystal_rot_2.composition)  # [1, 103]

        tmp_dist1 = {}
        tmp_dist1['atom_fea'] = atom_fea_rot_1
        tmp_dist1['nbr_fea'] = nbr_fea_rot_1
        tmp_dist1['nbr_fea_idx'] = nbr_fea_idx_rot_1
        tmp_dist1['groups'] = groups1
        tmp_dist1['enc_compo'] = enc_compo1
        tmp_dist1['coordinates'] = coordinates1
        tmp_dist1['target'] = target
        tmp_dist1['cif_id'] = cif_id
        tmp_dist1['atom_id'] = [crystal_rot_1[i].specie for i in range(len(crystal_rot_1))]

        tmp_dist2 = {}
        tmp_dist2['atom_fea'] = atom_fea_rot_2
        tmp_dist2['nbr_fea'] = nbr_fea_rot_2
        tmp_dist2['nbr_fea_idx'] = nbr_fea_idx_rot_2
        tmp_dist2['groups'] = groups2
        tmp_dist2['enc_compo'] = enc_compo2
        tmp_dist2['coordinates'] = coordinates2
        tmp_dist2['target'] = target
        tmp_dist2['cif_id'] = cif_id
        tmp_dist2['atom_id'] = [crystal_rot_2[i].specie for i in range(len(crystal_rot_2))]

        if self.args.data_src == 'binned_dos_128':
            pa = '../Mat2Spec_DATA/materials_with_edos_processed/'
            mkdirs(pa)
            torch.save(tmp_dist1, pa + "tmp_dist1_" + cif_id + '.chkpt')
            torch.save(tmp_dist2, pa + "tmp_dist2_" + cif_id + '.chkpt')
        elif self.args.data_src == 'ph_dos_51':
            pa = '../Mat2Spec_DATA/materials_with_phdos_processed/'
            mkdirs(pa)
            torch.save(tmp_dist1, pa + "tmp_dist1_" + cif_id + '.chkpt')
            torch.save(tmp_dist2, pa + "tmp_dist2_" + cif_id + '.chkpt')
        elif self.args.data_src == 'no_label_128':
            pa = '../Mat2Spec_DATA/materials_without_dos_processed/'
            mkdirs(pa)
            torch.save(tmp_dist1, pa + "tmp_dist1_" + cif_id + '.chkpt')
            torch.save(tmp_dist2, pa + "tmp_dist2_" + cif_id + '.chkpt')

        return [(atom_fea_rot_1, nbr_fea_rot_1, nbr_fea_idx_rot_1), groups1, enc_compo1, coordinates1, target, cif_id, [crystal_rot_1[i].specie for i in range(len(crystal_rot_1))]], [(atom_fea_rot_2, nbr_fea_rot_2, nbr_fea_idx_rot_2), groups2, enc_compo2, coordinates2, target, cif_id, [crystal_rot_2[i].specie for i in range(len(crystal_rot_2))]]

    def format_adj_matrix(self, adj_matrix):
        size = len(adj_matrix)
        src_list = list(range(size))
        all_src_nodes = torch.tensor([[x] * adj_matrix.shape[1] for x in src_list]).view(-1).long().unsqueeze(0)
        all_dst_nodes = adj_matrix.view(-1).unsqueeze(0)

        return torch.cat((all_src_nodes, all_dst_nodes), dim=0)
