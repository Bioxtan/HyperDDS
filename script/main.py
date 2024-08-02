import numpy as np
import pandas as pd
import torch

import warnings
import argparse
import yaml
from process_data import *
from utils import *
import sys
from ipdb import set_trace
import os
from sklearn.model_selection import KFold
import torch.utils.data as Data

from model import *


def data_split(synergy, rd_seed=0):
    synergy_pos = pd.DataFrame([i for i in synergy if i[3] == 1])
    synergy_neg = pd.DataFrame([i for i in synergy if i[3] == 0])
    train_size = 1
    synergy_cv_pos, synergy_test_pos = np.split(np.array(synergy_pos.sample(frac=1, random_state=rd_seed)),
                                                [int(train_size * len(synergy_pos))])
    synergy_cv_neg, synergy_test_neg = np.split(np.array(synergy_neg.sample(frac=1,random_state=rd_seed)),
                                                [int(train_size * len(synergy_neg))])
    
    synergy_cv_data = np.concatenate((np.array(synergy_cv_neg), np.array(synergy_cv_pos)), axis = 0)
    np.random.shuffle(synergy_cv_data)
    return synergy_cv_data

def train(drug_fea_set, cline_fea_set, synergy_adj, index, label):
    loss_train = 0
    true_ls, pred_ls = [],[]
    optimizer.zero_grad()
    for batch, (drug,cline) in enumerate(zip(drug_fea_set,cline_fea_set)):
        pred = model(drug.x, drug.edge_index, drug.batch, cline[0], synergy_adj,
                     index[:,0],index[:,1],index[:,2])
        loss = loss_func(pred,label)
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
        true_ls += label_train.cpu().detach().numpy().tolist()
        pred_ls += pred.cpu().detach().numpy().tolist()
    auc_train, aupr_train, f1_train, acc_train = metrics_graph(true_ls, pred_ls)
    return [auc_train, aupr_train, f1_train, acc_train],loss_train


def test(drug_fea_set, cline_fea_set, synergy_adj, index, label):
    model.eval()
    with torch.no_grad():
        for batch, (drug, cline) in enumerate(zip(drug_fea_set, cline_fea_set)):
            pred = model(drug.x, drug.edge_index, drug.batch, cline[0],synergy_adj,
                                              index[:, 0], index[:, 1], index[:, 2])
        loss = loss_func(pred, label)
        auc_test, aupr_test, f1_test, acc_test = metrics_graph(label.cpu().detach().numpy(),
                                                               pred.cpu().detach().numpy())
        return [auc_test, aupr_test, f1_test, acc_test], loss.item(), pred.cpu().detach().numpy()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(description='implementation of Hypersynergy')
    
    parser.add_argument('-e','--epoch',type=int,default=2000)
    parser.add_argument('-lr','--learningrate',type=float,default=1e-5,help='initial learning rate for adam')
    parser.add_argument('-s','--seed',type=int,default=0)

    args = parser.parse_args()


    seed = args.seed
    epochs = args.epoch
    learning_rate = args.learningrate
    L2 = 1e-4

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 构建文件的绝对路径
    drug_synergy_file = os.path.join(current_dir, '../data/triple.csv')
    cell_features_file = os.path.join(current_dir, '../data/allfeature.json')
    path = os.path.join(current_dir,'../result/')
    model_path = os.path.join(current_dir,'../trained_model/best_model.pth')
    seed_all(seed)

    drug_feature,cline_fea,synergy_data = getData(drug_synergy_file,cell_features_file)
    

    drug_set = Data.DataLoader(dataset=GraphDataset(graphs_dict=drug_feature), 
                               collate_fn=collate,batch_size = len(drug_feature), shuffle=False )
    cline_fea = torch.from_numpy(cline_fea).to(device)
    cline_set = Data.DataLoader(dataset=Data.TensorDataset(cline_fea),
                                batch_size=len(cline_fea),shuffle=False)
    
    cv_data = data_split(synergy_data)

    final_metric = np.zeros(4)
    fold_num = 0
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    for train_index, validation_index in kf.split(cv_data):
        #construct train,validation,test set.
        synergy_train, synergy_validation = cv_data[train_index], cv_data[validation_index]
        s = 0.5
        split_index = int(s * len(synergy_validation))
        synergy_validation, synergy_test = np.split(synergy_validation,[split_index])
        index_test = torch.from_numpy(synergy_test).to(device)
        label_test = torch.from_numpy(np.array(synergy_test[:, 3], dtype = 'float32')).to(device)
        #np.savetxt(path + 'test_' + str(fold_num) + '_true.txt', synergy_test[:,3])
        #np.savetxt(path + 'val_' + str(fold_num) + '_true.txt', synergy_validation[:, 3])
        label_train = torch.from_numpy(np.array(synergy_train[:, 3], dtype='float32')).to(device)
        label_validation = torch.from_numpy(np.array(synergy_validation[:, 3], dtype='float32')).to(device)
        index_train = torch.from_numpy(synergy_train).to(device)
        index_validation = torch.from_numpy(synergy_validation).to(device)

        #construct hypergraph set
        edge_data = synergy_train[synergy_train[:,3] == 1, 0:3]
        synergy_edge = edge_data.reshape(1,-1)  #reshape为只有1行的二维数组
        index_num = np.expand_dims(np.arange(len(edge_data)), axis = -1)

        synergy_num = np.concatenate((index_num,index_num,index_num),axis = 1)
        synergy_num = np.array(synergy_num).reshape(1,-1)
        synergy_graph = np.concatenate((synergy_edge,synergy_num), axis = 0)
        synergy_graph = torch.from_numpy(synergy_graph).type(torch.LongTensor).to(device)

        #model_build

        model = Hypersynergy(Encoder(dim_drug = 75, dim_cellline = cline_fea.shape[-1], hidden=128,output = 100),
                             HgnnEncoder(in_channels = 100, out_channels = 256), Decoder(in_channels = 1024)).to(device)
        loss_func = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate,weight_decay = L2)
        
        best_metric = [0,0,0,0]
        best_epoch = 0
        for epoch in range(epochs):
            model.train()
            train_metric,train_loss = train(drug_set, cline_set, synergy_graph, index_train, label_train)
            val_metric, val_loss, _ = test(drug_set, cline_set, synergy_graph, index_validation, label_validation)

            if epoch % 20 == 0:
                print('Epoch: {:05d},'.format(epoch), 'loss_train: {:.6f},'.format(train_loss),
                          'AUC: {:.6f},'.format(train_metric[0]), 'AUPR: {:.6f},'.format(train_metric[1]),
                          'F1: {:.6f},'.format(train_metric[2]), 'ACC: {:.6f},'.format(train_metric[3]),
                          )
                print('Epoch: {:05d},'.format(epoch), 'loss_val: {:.6f},'.format(val_loss),
                          'AUC: {:.6f},'.format(val_metric[0]), 'AUPR: {:.6f},'.format(val_metric[1]),
                          'F1: {:.6f},'.format(val_metric[2]), 'ACC: {:.6f},'.format(val_metric[3]))
            if val_metric[0] > best_metric[0]:
                best_metric = val_metric
                best_epoch = epoch
                torch.save(model.state_dict(), model_path)
        print('The best results on validation set, Epoch: {:05d},'.format(best_epoch),
                  'AUC: {:.6f},'.format(best_metric[0]),
                  'AUPR: {:.6f},'.format(best_metric[1]), 'F1: {:.6f},'.format(best_metric[2]),
                  'ACC: {:.6f},'.format(best_metric[3]))
        
        model.load_state_dict(torch.load(model_path))
        
        val_metric, _, y_val_pred = test(drug_set, cline_set, synergy_graph, index_validation, label_validation,
                                             )
        test_metric, _, y_test_pred = test(drug_set, cline_set, synergy_graph, index_test, label_test)

        np.savetxt(path + 'val_' + str(fold_num) + '_pred.txt', y_val_pred)
        np.savetxt(path + 'test_' + str(fold_num) + '_pred.txt', y_test_pred)

        print('test results, AUC: {:.6f},'.format(test_metric[0]),
              'AUPR: {:.6f},'.format(test_metric[1]),
              'F1: {:.6f},'.format(test_metric[2]), 'ACC: {:.6f},'.format(test_metric[3]))
        final_metric += test_metric
        fold_num = fold_num + 1
    final_metric /= 5
    print('Final 5-cv average results, AUC: {:.6f},'.format(final_metric[0]),
              'AUPR: {:.6f},'.format(final_metric[1]),
              'F1: {:.6f},'.format(final_metric[2]), 'ACC: {:.6f},'.format(final_metric[3]))






