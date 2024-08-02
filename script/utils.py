from rdkit import DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem import AllChem as Chem
import numpy as np
import pandas as pd
import torch
import random
import yaml
import os
from easydict import EasyDict
from torch_geometric.data import InMemoryDataset,Batch
from torch_geometric import data as DATA
from sklearn.metrics import roc_auc_score, precision_recall_curve, mean_squared_error, r2_score

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def seed_all(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)



def get_MACCS(smiles):
    m = Chem.MolFromSmiles(smiles)
    arr = np.zeros((1,), np.float32)
    fp = MACCSkeys.GenMACCSKeys(m)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


# -----molecular_graph_feature
def calculate_graph_feat(feat_mat, adj_list):
    assert feat_mat.shape[0] == len(adj_list)
    adj_mat = np.zeros((len(adj_list), len(adj_list)), dtype='float32')
    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for each in nodes:
            adj_mat[i, int(each)] = 1
    assert np.allclose(adj_mat, adj_mat.T)
    x, y = np.where(adj_mat == 1)
    adj_index = np.array(np.vstack((x, y)))
    return [feat_mat, adj_index]


def drug_feature_extract(drug_data):
    drug_data = pd.DataFrame(drug_data).T
    drug_feat = [[] for _ in range(len(drug_data))]
    for i in range(len(drug_feat)):
        feat_mat, adj_list = drug_data.iloc[i]
        drug_feat[i] = calculate_graph_feat(feat_mat, adj_list)
    return drug_feat



class GraphDataset(InMemoryDataset):
    def __init__(self,root='/home/xy_th/hs/data',dataset = 'syn',transform = None, pre_transform = None, graphs_dict = None, dttype = None ):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        
        self.dttype = dttype
        self.process(graphs_dict)
    
    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + f'_data_{self.dttype}.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def process(self, graphs_dict):
        data_list = []
        for data_mol in graphs_dict:
            features = torch.Tensor(data_mol[0]).to(device)
            edge_index = torch.LongTensor(data_mol[1]).to(device)
            GCNdata = DATA.Data(x=features, edge_index=edge_index)
            data_list.append(GCNdata)
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        return self.data[idx]

def collate(data_list):
    batchA = Batch.from_data_list([data for data in data_list])
    return batchA.to(device)
        


def metrics_graph(yt, yp):
    precision, recall, _, = precision_recall_curve(yt, yp)
    aupr = -np.trapz(precision, recall)
    auc = roc_auc_score(yt, yp)
    # ---f1,acc,recall, specificity, precision
    real_score = np.mat(yt)
    predict_score = np.mat(yp)
    sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(sorted_predict_score_num * np.arange(1, 1000) / 1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]
    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN
    tpr = TP / (TP + FN)
    recall_list = tpr
    precision_list = TP / (TP + FP)
    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)
    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return auc, aupr, f1_score[0, 0], accuracy[0, 0]  # , recall[0, 0], specificity[0, 0], precision[0, 0]