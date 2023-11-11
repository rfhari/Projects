import torch
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

torch.manual_seed(0)  
np.random.seed(0)
  
EPOCH = 150
BATCH_SIZE = 64
LR = 0.001        
DOWNLOAD_MNIST = False
N_TEST_IMG = 5

class MFIDataset(Dataset):
    def __init__(self, x, y, c):
        super().__init__()
        assert len(x) == len(y)
        self._x = x
        self._y = y
        self._c = c
    
    def __len__(self):
        return len(self._x)
      
    def __getitem__(self, index):
        x_item = self._x[index]
        y_item= self._y[index]
        class_item = self._c[index]
#         features = one_hot_fea(x_item)
        return x_item, y_item, class_item, index

def morgan_featurizer(mols, labels):
    mol_feat, lab = [], []
    for i, mol in enumerate(mols):
        feat2, _ = morgan_featurization(mol, labels[i])
        mol_feat.append(feat2.reshape(1024))
        lab.append(labels[i])
    mol_feat = np.array(mol_feat)
    print(np.shape(mol_feat))
    labels = np.array(lab).reshape(-1, 1)
    return mol_feat, labels

# def load_all():
#     all_smiles, all_lab = [], []
#     with open('NR-AR_all.smiles') as f:
#         for l in f:
#             p = l.strip().split(" ")
#             all_smiles.append(l.strip().split(" ")[0])
#             all_lab.append(int(l.strip().split(" ")[1]))
#     all_smiles = np.array(all_smiles)
#     all_lab = np.array(all_lab)
#     valid_smiles = [Chem.MolFromSmiles(sm) for sm in all_smiles if Chem.MolFromSmiles(sm)]
#     valid_labels = [all_lab[i] for i, sm in enumerate(all_smiles) if Chem.MolFromSmiles(sm)]
#     return valid_smiles, valid_labels

def load_all():
    f = pd.read_csv("delaney-processed modified.csv")
    all_smiles = np.array(f['smiles'])
    all_lab = np.array(f['labels'])
    valid_mols = [Chem.MolFromSmiles(sm) for sm in all_smiles if Chem.MolFromSmiles(sm) is not None]
    valid_labels = [all_lab[i] for i, sm in enumerate(all_smiles) if Chem.MolFromSmiles(sm) is not None]
    return valid_mols, valid_labels

x_train0, y_train0 = load_all()
train_data = []

# l1 = round(0.8 * len(x_train0))
# l2 = l1 + round(0.1 * len(x_train0))

# X_train, y_train = x_train0[:l1], y_train0[:l1]
# X_val, y_val = x_train0[l1:l2], y_train0[l1:l2]

class_ = []
for i in range (len(y_train0)):
    if y_train0[i] <= -6:
        class_.append(0)
    elif y_train0[i] <= -4:
        class_.append(1)
    elif y_train0[i] <=-2:
        class_.append(2)
    elif y_train0[i] <= 0:
        class_.append(3)    
    else:
        class_.append(4)    

X_train, y_train = morgan_featurizer(x_train0, y_train0)
data = MFIDataset(X_train, y_train, class_)
train_size = int(0.8 * len(data))
test_size = len(data) - train_size
train_set, test_set = torch.utils.data.random_split(data, [train_size, test_size])
train_loader = DataLoader(train_set, batch_size=32, shuffle=True, pin_memory=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, pin_memory=True, num_workers=0)

# ytrain_bins = []
# for i in y_train:
#     if i <=2.57:
#         ytrain_bins.append(0)
#     # elif i<=3:
#         # ytrain_bins.append(1)       
#     # elif i<=3.3:
#     #     ytrain_bins.append(2)
#     else:
#         ytrain_bins.append(1)

# yval_bins = []
# for i in y_val:
#     if i <=2.57:
#         yval_bins.append(0)
#     # elif i<=3:
#         # yval_bins.append(1)
#     # elif i<=3.3:
#     #     yval_bins.append(2)
#     else:
#         yval_bins.append(1)
    
# for i in range(len(X_train)):
#     train_data.append([X_train[i], ytrain_bins[i]])
    
# train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, num_workers=8)

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            # nn.Linear(8192, 4096),
            # nn.ReLU(),
            # nn.Linear(4096, 2048),
            # nn.ReLU(),
            # nn.Linear(2048, 1024),
            # nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8), 
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(), 
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(), 
            nn.Linear(256, 512),
            nn.ReLU(), 
            nn.Linear(512, 1024),
            nn.Sigmoid(), 
            # nn.Linear(1024, 2048),
            # nn.ReLU(),
            # nn.Linear(2048, 4096),
            # nn.ReLU(),
            # nn.Linear(4096, 8192),
            # nn.ReLU(),
        )

        self.dense = nn.Sequential(
            nn.Linear(2, 5),
            )
    
    def forward(self, x):
        encoded = self.encoder(x)
        predictor = self.dense(encoded)
        decoded = self.decoder(encoded)
        return encoded, predictor, decoded

autoencoder = AutoEncoder()
autoencoder = autoencoder.float().cuda()

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
construction_loss = nn.MSELoss()
classification_loss = nn.CrossEntropyLoss()

def val(autoencoder):
    autoencoder.eval()
    with torch.no_grad():
        loss_full = []
        all_val_encoder_output = []
        for step, (x, y, b_label, train_index) in enumerate (test_loader):
            b_x = x.view(-1, 1024)   
            b_y = x.view(-1, 1024)   
            b_x, b_y, b_label = b_x.cuda(), b_y.cuda(), b_label.cuda()
           
            encoded, predict, decoded = autoencoder(b_x.float())
            loss1 = construction_loss(b_x.float(), decoded) 
            loss2 = classification_loss(predict, b_label)
            loss = loss1 + loss2
            # loss = loss1

            all_val_encoder_output.append(encoded.detach().cpu().numpy())
            loss_full.append(loss.item())
        return all_val_encoder_output, np.array(loss_full).mean()

for epoch in range(EPOCH):
    train_loss, train_latent, L1, L2, class_epoch, indices = [], [], [], [], [], []
    for step, (x, y, b_label, train_index) in enumerate(train_loader):
        b_x = x.view(-1, 1024)   
        b_y = x.view(-1, 1024)   
        b_x, b_y, b_label = b_x.cuda(), b_y.cuda(), b_label.cuda()
        encoded, predict, decoded = autoencoder(b_x.float())
        _, predict_ = torch.max(predict.data, 1)
        # if predict_==1:
        # print("predict:", predict_)
        # print("ground truth:", b_label)
        loss1 = construction_loss(b_x.float(), decoded)
        loss2 = classification_loss(predict, b_label)
        loss = loss1 + loss2 #WITH CROSS ENTROPY LOSS
        # loss = loss1 # WITHOUT CROSS ENTROPY LOSS
        optimizer.zero_grad()               
        loss.backward()                     
        optimizer.step()
        class_epoch.append(predict_.detach().cpu().numpy())
        indices.append(train_index.cpu().numpy())
        train_loss.append(loss.item())
        L1.append(loss1.item())
        L2.append(loss2.item())
        train_latent.append(encoded.detach().cpu().numpy())
    train_loss_avg = np.array(train_loss).mean()
    l1_avg = np.array(L1).mean()
    l2_avg = np.array(L2).mean()
    val_encoder_output, valid_loss = val(autoencoder)
    loss_avg = train_loss_avg
    if epoch==0:
        best_loss = loss_avg     
        class_epoch_v = np.hstack(np.asarray(class_epoch))
        index_values = np.hstack(np.asarray(indices))
        best_encoder_output = np.vstack(np.asarray(val_encoder_output))
        best_encoder_output_train = np.vstack(np.asarray(train_latent)) 
        print(best_encoder_output.shape)
    
    elif best_loss > loss_avg:
        best_loss = loss_avg
        index_values = np.hstack(np.asarray(indices))
        class_epoch_v = np.hstack(np.asarray(class_epoch))
        best_encoder_output = np.vstack(np.asarray(val_encoder_output))
        best_encoder_output_train = np.vstack(np.asarray(train_latent))
        e = epoch
        info = {'epoch': epoch+1,
                'model': autoencoder.state_dict(),
                'best_acc1': loss_avg,
                'optimizer' : optimizer.state_dict()
                }
    print('epoch {}, train loss {}, val loss {}, l1 {}, l2 {}'.format(epoch, train_loss_avg, valid_loss, l1_avg, l2_avg))

print("best loss:", best_loss, "best epoch:", e)    

x, y = best_encoder_output_train[:, 0], best_encoder_output_train[:, 1]
c_ = class_epoch_v
np.save("x-coord-esol", x)
np.save("y-coord-esol", y)
np.save("c_coord-esol", c_)
fig, ax = plt.subplots()
scatter = ax.scatter(x, y, c = c_, label = c_)
legend1 = ax.legend(*scatter.legend_elements(), title = 'Classes')
plt.xlabel('Latent Vector1')
plt.ylabel('Latent Vector2')
plt.show()

m0, m1, m2, m3, m4 = [], [], [], [], []

for i,k in enumerate(c_):
    if k == 0:
        m0.append(x_train0[index_values[i]])
    elif k == 1:
        m1.append(x_train0[index_values[i]])
    elif k == 2:
        m2.append(x_train0[index_values[i]])
    elif k == 3:
        m3.append(x_train0[index_values[i]])    
    else:
        m4.append(x_train0[index_values[i]])    

# q = np.random.randint(0, len(m4))
# q_m = m4[q]
# fps = AllChem.GetMorganFingerprintAsBitVect(q_m, 4, useFeatures=True, nBits = 1024)
# array = np.zeros((1,));
# DataStructs.ConvertToNumpyArray(fps, array);

# inp = torch.from_numpy(array)
# o1, o2, o3 = autoencoder(inp.cuda().float())

# noise = np.random.normal(0, 1, size=(100, 2))
# on = o1.view(-1, 2) + torch.from_numpy(noise).cuda()

x_ = np.linspace(np.min(x), np.max(x), 10)
y_ = np.linspace(np.min(y), np.max(y), 10)
cp = np.array(np.meshgrid(x_, y_)).T.reshape(-1, 2)
cp_v = torch.from_numpy(cp).cuda()

output_ = []
for k in cp_v:
    k = k.reshape(-1, 2)
    lp = autoencoder.decoder(k.cuda().float())
    lp_ = lp.detach().data.cpu().numpy()
    output_.append(lp_)

output_a = np.asarray(output_).reshape(-1, 1024)    
np.save("smiles_temporal_esol_rand.npy", output_a)