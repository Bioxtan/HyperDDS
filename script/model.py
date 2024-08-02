import torch
import torch.nn as nn
from torch_geometric.nn import HypergraphConv,GCNConv,global_max_pool,global_mean_pool
from ipdb import set_trace
from attention import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


drug_num = 38
cline_num = 31

class Encoder(nn.Module):
    def __init__(self,dim_drug,dim_cellline,hidden,output,use_GMP=True):
        super(Encoder,self).__init__()
        #encode drug using GCN
        self.conv1 = GCNConv(dim_drug,hidden)
        self.batch_conv1 = nn.BatchNorm1d(hidden)
        self.conv2 = GCNConv(hidden,output)
        self.batch_conv2 = nn.BatchNorm1d(output)

        #encode cell line
        self.fc_cell1 = nn.Linear(dim_cellline,256)
        self.batch_cell1 = nn.BatchNorm1d(256)
        self.fc_cell2 = nn.Linear(256,output)
        self.reset_para()
        self.act = nn.ReLU()
        self.use_GMP=use_GMP

    def reset_para(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return
    
    def forward(self,drug_feature,drug_adj,ibatch,gexpr_data):
        
        x_drug = self.conv1(drug_feature, drug_adj)
        x_drug = self.batch_conv1(self.act(x_drug))
        x_drug = self.conv2(x_drug, drug_adj)
        x_drug = self.act(x_drug)
        x_drug = self.batch_conv2(x_drug)

        if self.use_GMP:
            x_drug = global_max_pool(x_drug, ibatch)
        else:
            x_drug = global_mean_pool(x_drug, ibatch)

        # Cell line encoding
        x_cellline = self.act(self.fc_cell1(gexpr_data))
        x_cellline = self.batch_cell1(x_cellline)
        x_cellline = self.act(self.fc_cell2(x_cellline))

        return x_drug, x_cellline

class Decoder(torch.nn.Module):
    def __init__(self, in_channels):
        super(Decoder, self).__init__()

        self.Drug_cell_mutual = Drug_cell_mutual_attention(256,256,4,0.1,0.1)
        self.drug_self_attention = Drug_self_attention(256,256,4,0.1,0.1)
        self.cell_self_attention = Cell_self_attention(256,256,4,0.1,0.1)
        self.drug_drug_mutual_attention = Drug_cross_attention(256,256,4,0.1,0.1)
        self.cell_cell_mutual_attention = Cell_mutual_attention(256,256,4,0.1,0.1)

        self.fc1 = nn.Linear(in_channels, in_channels // 2)
        self.batch1 = nn.BatchNorm1d(in_channels // 2)
        self.fc2 = nn.Linear(in_channels // 2, in_channels // 4)
        self.batch2 = nn.BatchNorm1d(in_channels // 4)
        self.fc3 = nn.Linear(in_channels // 4, 1)

        self.reset_parameters()
        self.drop_out = nn.Dropout(0.4)
        self.act = nn.Tanh()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, graph_embed, druga_id, drugb_id, cellline_id):
        cell_r_drugA,drugA_r_cell,a1,b1 = self.Drug_cell_mutual(graph_embed[cellline_id, :],graph_embed[druga_id, :])
        cell_r_drugB,drugB_r_cell,a2,b2 = self.Drug_cell_mutual(graph_embed[cellline_id, :],graph_embed[drugb_id, :])
        cell_r_drugA,cell_r_drugA_p = self.cell_self_attention(cell_r_drugA)
        cell_r_drugB,cell_r_drugB_p= self.cell_self_attention(cell_r_drugB)
        drugA_r_cell,drugA_attention_probability = self.drug_self_attention(drugA_r_cell)
        drugB_r_cell,drugB_attention_probability = self.drug_self_attention(drugB_r_cell)
        drug1,drug2,c,d= self.drug_drug_mutual_attention(drugA_r_cell,drugB_r_cell)
        cell1,cell2,e,f = self.cell_cell_mutual_attention(cell_r_drugA,cell_r_drugB)

        h1 = torch.cat((drug1, drug2, cell1,cell2), 1)
        
        h = self.act(self.fc1(h1))
        h = self.batch1(h)
        h = self.drop_out(h)
        h = self.act(self.fc2(h))
        h = self.batch2(h)
        h = self.drop_out(h)
        h = self.fc3(h)
        return torch.sigmoid(h.squeeze(dim=1))
    

class HgnnEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HgnnEncoder, self).__init__()
        self.conv1 = HypergraphConv(in_channels, 256)
        self.batch1 = nn.BatchNorm1d(256)
        self.conv2 = HypergraphConv(256, 256)
        self.batch2 = nn.BatchNorm1d(256)
        self.conv3 = HypergraphConv(256, out_channels)
        self.act = nn.ReLU()

    def forward(self, x, edge):
        x = self.batch1(self.act(self.conv1(x, edge)))
        x = self.batch2(self.act(self.conv2(x, edge)))
        x = self.act(self.conv3(x, edge))
        return x


class Hypersynergy(torch.nn.Module):
    def __init__(self,Encoder,HgnnEncoder,Decoder):
        super(Hypersynergy,self).__init__()
        self.Encoder = Encoder
        self.graph_encoder = HgnnEncoder
        self.decoder = Decoder

    def forward(self, drug_feature, drug_adj, ibatch, gexpr_data, adj, druga_id, drugb_id, cellline_id):
        drug_embed, cellline_embed = self.Encoder(drug_feature, drug_adj, ibatch, gexpr_data)
        merge_embed = torch.cat((drug_embed, cellline_embed), 0)
        graph_embed = self.graph_encoder(merge_embed, adj)
        res = self.decoder(graph_embed, druga_id, drugb_id, cellline_id)
        return res