import torch
from torch import nn
import torch.nn.functional as F
import math


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):

        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        # Normalize input_tensor
        s = (x - u).pow(2).mean(-1, keepdim=True)
        # Apply scaling and bias
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class SelfAttention(nn.Module):
    def __init__(self,hidden_size,num_attention_heads,dropout_rate):     #隐藏层，注意力头数，dropout
        super(SelfAttention,self).__init__()
        if hidden_size % num_attention_heads !=0:
            raise ValueError(
                "The hidden size(%d) is not a multiple  of the number of attention"
                "heads(%d)" % (hidden_size,num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size,self.all_head_size)
        self.key = nn.Linear(hidden_size,self.all_head_size)
        self.value = nn.Linear(hidden_size,self.all_head_size)

        self.dropout = nn.Dropout(dropout_rate)

    def transpose_for_scores(self,x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0,2,1,3)
    
    def forward(self,input_tensor,attention_mask=None):
        input_tensor = input_tensor.unsqueeze(1)       #由于drug的embedding的size是二维的，(batch_size,384)，将其变成(batch_size,1,384)
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer,key_layer.transpose(-1,-2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probability = nn.Softmax(dim=-1)(attention_scores)    #将注意力分数转换为概率分布，dim=-1表示选择最后一个维度进行softmax操作
        attention_probability = self.dropout(attention_probability)
        
        context_layer = torch.matmul(attention_probability,value_layer)
        context_layer = context_layer.permute(0,2,1,3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = context_layer.squeeze(1)
        return context_layer,attention_probability
    


class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_dropout_rate):
        super(CrossAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_dropout_rate)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, drugA, drugB, drugA_attention_mask=None):
        # update drugA
        drugA = drugA.unsqueeze(1)
        drugB = drugB.unsqueeze(1)
        mixed_query_layer = self.query(drugA)
        mixed_key_layer = self.key(drugB)
        mixed_value_layer = self.value(drugB)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if drugA_attention_mask == None:
            attention_scores = attention_scores
        else:
            attention_scores = attention_scores + drugA_attention_mask

        attention_probs_0 = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs_0)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape) 
        context_layer = context_layer.squeeze(1)      

        return context_layer, attention_probs_0

class Intermediate(nn.Module): #定义一个中间层
    def __init__(self,hidden_size,intermediate_size):
        super(Intermediate,self).__init__()
        self.dense = nn.Linear(hidden_size,intermediate_size)
    
    def forward(self,hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states
    
class Output(nn.Module):  #输出层
    def __init__(self,intermediate_size,hidden_size,hidden_dropout_rate):
        super(Output,self).__init__()
        self.dense = nn.Linear(intermediate_size,hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_rate)
    def forward(self,hidden_states,input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states+input_tensor)
        return hidden_states
    
# Drug self-attention encoder
class Drug_self_attention(nn.Module):
    def __init__(self,hidden_size,intermediate_size,num_attention_heads,attention_dropout_rate,hidden_dropout_rate):
        super(Drug_self_attention,self).__init__()
        self.attention = SelfAttention(hidden_size,num_attention_heads,attention_dropout_rate)
        self.LayerNorm = LayerNorm(hidden_size)
        self.layers = nn.Sequential(
            nn.Linear(hidden_size,hidden_size),
            nn.Dropout(hidden_dropout_rate),
        )
        self.intermediate = Intermediate(hidden_size,intermediate_size)
        self.output = Output(intermediate_size,hidden_size,hidden_dropout_rate)
    def forward(self,input_tensor,attention_mask=None):
        attention_output,attention_probability = self.attention(input_tensor,attention_mask)
        attention_output = self.layers(attention_output)
        attention_output = self.LayerNorm(attention_output+input_tensor)
        Intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(Intermediate_output,attention_output)
        return layer_output,attention_probability
    
#test
""" drug  = Drug_self_attention(384,256,8,0.1,0.1)
drug1 = torch.randn(128,384)
b,c = drug(drug1)
print(b) """

# Cell self-attention encoder
class Cell_self_attention(nn.Module):
    def __init__(self,hidden_size,intermediate_size,num_attention_heads,attention_dropout_rate,hidden_dropout_rate):
        super(Cell_self_attention,self).__init__()
        self.attention = SelfAttention(hidden_size,num_attention_heads,attention_dropout_rate)
        self.LayerNorm = LayerNorm(hidden_size)
        self.layers = nn.Sequential(
            nn.Linear(hidden_size,hidden_size),
            nn.Dropout(hidden_dropout_rate)
        )
        self.dense = nn.Sequential(
            nn.Linear(hidden_size,intermediate_size),
            nn.ReLU(),
            nn.Dropout(hidden_dropout_rate),
            nn.Linear(intermediate_size,hidden_size)
        )
    def forward(self,cell_lines,attention_mask=None):
        cell_lines = self.LayerNorm(cell_lines)
        attention_output,attention_probability = self.attention(cell_lines,attention_mask)
        cell_lines_2 = cell_lines + attention_output
        cell_lines_3 = self.LayerNorm(cell_lines_2)

        cell_lines_4 = self.dense(cell_lines_3)

        layer_output = cell_lines_2 + cell_lines_4

        return layer_output,attention_probability

#test
""" cell = Cell_self_attention(954,1908,9,0.1,0.1)
cell_line = torch.randn(128,954)
b,c = cell(cell_line)
print(b) """

class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states    


# Drug-drug mutual attention module
class Drug_cross_attention(nn.Module):
    def __init__(self,hidden_size,intermediate_size,num_attention_heads,attention_dropout_rate,hidden_dropout_rate):
        super(Drug_cross_attention,self).__init__()
        self.attention = CrossAttention(hidden_size,num_attention_heads,attention_dropout_rate)
        self.intermediate = Intermediate(hidden_size,intermediate_size)
        self.selfoutput = SelfOutput(hidden_size,hidden_dropout_rate)
        self.output = Output(intermediate_size,hidden_size,hidden_dropout_rate)

    def forward(self,drugA,drugB):
        drugA_self_output,drugA_probability = self.attention(drugA,drugB)
        drugB_self_output,drugB_probability = self.attention(drugB,drugA)
        drugA_attention_output = self.selfoutput(drugA_self_output,drugA)
        drugB_attention_output = self.selfoutput(drugB_self_output,drugB)
        drugA_intermediate_output = self.intermediate(drugA_attention_output)
        drugA_layer_output = self.output(drugA_intermediate_output, drugA_attention_output)
        drugB_intermediate_output = self.intermediate(drugB_attention_output)
        drugB_layer_output = self.output(drugB_intermediate_output, drugB_attention_output)
        return drugA_layer_output, drugB_layer_output, drugA_probability, drugB_probability

# test
""" drug_cross_attention = Drug_cross_attention(384,256,8,0.1,0.1)
drug1 = torch.randn(128,384)
drug2 = torch.randn(128,384)
a,b,c,d = drug_cross_attention(drug1,drug2)
print(a)
 """

 # Cell-cell mutual-attention encoder
class Cell_mutual_attention(nn.Module):
    def __init__(self,hidden_size,intermediate_size,num_attention_heads,attention_dropout_rate,hidden_dropout_rate):
        super(Cell_mutual_attention,self).__init__()
        self.LayerNorm = LayerNorm(hidden_size)
        self.attention = CrossAttention(hidden_size,num_attention_heads,attention_dropout_rate)
        self.dense = nn.Sequential(
            nn.Linear(hidden_size,intermediate_size),
            nn.ReLU(),
            nn.Dropout(hidden_dropout_rate),
            nn.Linear(intermediate_size,hidden_size)
        )
        self.selfoutput =  SelfOutput(hidden_size,hidden_dropout_rate)

    def forward(self,cellA,cellB):
        cellA_1 = self.LayerNorm(cellA)
        cellB_1 = self.LayerNorm(cellB)

        cellA_self_output,cellA_probability = self.attention(cellA,cellB)
        cellB_self_output,cellB_probability = self.attention(cellB,cellA)
        cellA_attention_output = self.selfoutput(cellA_self_output,cellA)
        cellB_attention_output = self.selfoutput(cellB_self_output,cellB)

        #cellA_output
        cellA_2 = cellA_1 + cellA_attention_output
        cellA_3 = self.LayerNorm(cellA_2)
        cellA_4 = self.dense(cellA_3)
        cellA_layer_output = cellA_2 + cellA_4

        # cellB_output
        cellB_2 = cellB_1 + cellB_attention_output
        cellB_3 = self.LayerNorm(cellB_2)
        cellB_4 = self.dense(cellB_3)
        cellB_layer_output = cellB_2 + cellB_4

        return cellA_layer_output, cellB_layer_output, cellA_probability, cellB_probability
    
# test
""" cell_mu_attention = Cell_mutual_attention(954,256,6,0.1,0.1)
cell1 = torch.randn(128,954)
cell2 = torch.randn(128,954)
a,b,c,d = cell_mu_attention(cell1,cell2)
print(a)
 """

# Drug_cell_mutual_attention_module
class Drug_cell_mutual_attention(nn.Module):
    def __init__(self,hidden_size,intermediate_size,num_attention_heads,attention_dropout_rate,hidden_dropout_rate):
        super(Drug_cell_mutual_attention,self).__init__()
        self.LayerNorm = LayerNorm(hidden_size)
        self.attention = CrossAttention(hidden_size,num_attention_heads,attention_dropout_rate)
        self.intermediate = Intermediate(hidden_size,intermediate_size)
        self.output = Output(intermediate_size,hidden_size,hidden_dropout_rate)
        self.dense = nn.Sequential(
            nn.Linear(hidden_size,intermediate_size),
            nn.ReLU(),
            nn.Dropout(hidden_dropout_rate),
            nn.Linear(intermediate_size,hidden_size)
        )
        self.selfoutput = SelfOutput(hidden_size,hidden_dropout_rate)
    def forward(self,cell,drug):

        cell_1 = self.LayerNorm(cell)
        cell_self_out,cell_attention_probability = self.attention(cell,drug)
        drug_self_out,drug_attention_probability = self.attention(drug,cell)
        cell_attention_output = self.selfoutput(cell_self_out,cell)
        drug_attention_output = self.selfoutput(drug_self_out,drug)

        cell_2 = cell_1 +cell_attention_output
        cell_3 = self.LayerNorm(cell_2)
        cell_4 = self.dense(cell_3)
        cell_layer_output = cell_2 + cell_4

        drug_intermediate_output = self.intermediate(drug_attention_output)
        drug_layer_output = self.output(drug_intermediate_output,drug_attention_output)

        return cell_layer_output,drug_layer_output,cell_attention_probability,drug_attention_probability
#test
""" cell = torch.randn(128,954)
drug = torch.randn(128,954)
cell_drugmutual = Drug_cell_mutual_attention(954,512,9,0.1,0.1)
a,b,c,d = cell_drugmutual(cell,drug)
print(a) """

#