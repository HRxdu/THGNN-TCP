import dgl
import math
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureTransformLayer(nn.Module):
    def __init__(self, feature_dict, args):
        super(FeatureTransformLayer, self).__init__()
        feature_dim_list = [feature.shape[1] for feature in feature_dict.values()]
        hidden_dim = args.hidden_dim
        self.device = args.device
        self.feature_dict = feature_dict
        self.fc_list = nn.ModuleList([nn.Linear(feature_dim, args.embedding_dim, bias=True) for feature_dim in feature_dim_list])  # 0 for ipc, 1 for paper, 2 for patent
        #self.fc_list = nn.ModuleList([nn.Linear(feature_dim, 128, bias=True) for feature_dim in feature_dim_list]) # 0 for ipc, 1 for paper, 2 for patent
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

    def forward(self):
        ipc_feature = self.feature_dict['ipc'].to(torch.float32).to(self.device)
        patent_feature = self.feature_dict['patent'].to(torch.float32).to(self.device)
        paper_feature = self.feature_dict['paper'].to(torch.float32).to(self.device)

        ipc_vec = self.fc_list[0](ipc_feature)
        patent_vec = self.fc_list[2](patent_feature)
        paper_feature = self.fc_list[1](paper_feature)

        return torch.cat([ipc_vec, patent_vec, paper_feature], dim=0) # [num_ipc + num_patent + num_paper, out_dim]

class IntraMetapathAggregation(nn.Module):
    def __init__(self, args, mtype):
        super(IntraMetapathAggregation, self).__init__()
        self.device = args.device
        self.num_logits_heads = args.num_logits_heads
        self.mtype = mtype
        self.num_heads = args.num_heads
        self.hidden_dim = args.hidden_dim
        '''self.linear_i = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim, bias=False)
        self.linear_p = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim, bias=False)
        self.linear_pa = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim, bias=False)'''
        """self.linear_i = nn.Linear(in_features=128, out_features=self.hidden_dim, bias=False)
        self.linear_p = nn.Linear(in_features=128, out_features=self.hidden_dim, bias=False)
        self.linear_pa = nn.Linear(in_features=128, out_features=self.hidden_dim, bias=False)"""
        self.linear_i = nn.Linear(in_features=args.embedding_dim, out_features=self.hidden_dim, bias=False)
        self.linear_p = nn.Linear(in_features=args.embedding_dim, out_features=self.hidden_dim, bias=False)
        self.linear_pa = nn.Linear(in_features=args.embedding_dim, out_features=self.hidden_dim, bias=False)
        self.linear_talking_heads = nn.Linear(in_features=self.num_heads, out_features=self.num_logits_heads, bias=False)
        self.attn = nn.Parameter(torch.empty(size=(1, self.num_heads, self.hidden_dim)))
        self.dropout = nn.Dropout(args.dropout_rate)
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = edge_softmax
        self.fc_gate = nn.Linear(self.hidden_dim, 1, bias=True)
        nn.init.xavier_normal_(self.linear_i.weight, gain=1.414)
        nn.init.xavier_normal_(self.linear_p.weight, gain=1.414)
        nn.init.xavier_normal_(self.linear_pa.weight, gain=1.414)
        nn.init.xavier_normal_(self.attn, gain=1.414)
        nn.init.xavier_normal_(self.fc_gate.weight, gain=1.414)

    def positional_encoding(self, num_indices, max_seq_len, d_model):
        """
        生成位置编码矩阵
        :param max_seq_len: 序列最大长度
        :param d_model: 模型的维度
        :return: 位置编码矩阵，形状为(max_seq_len, d_model)
        """
        pe = torch.zeros(num_indices, max_seq_len, d_model)  # 初始化位置编码矩阵
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)  # 位置索引 (0, 1, ..., max_seq_len-1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # 计算衰减项

        pe[:, :, 0::2] = torch.sin(position * div_term)  # 偶数维度使用sin函数
        pe[:, :, 1::2] = torch.cos(position * div_term)  # 奇数维度使用cos函数

        return pe.to(self.device)

    def node_type_specific_agg(self, mtype: int, membd: torch.Tensor):
        if mtype == 0:
            '''max_seq_len = 3
            pe = self.positional_encoding(membd.shape[0], max_seq_len, self.hidden_dim)
            h_i_src = torch.squeeze(membd[:, 0, :] + pe[:, 0, :], dim=1)
            h_p = torch.squeeze(membd[:, 1, :] + pe[:, 1, :], dim=1)
            h_i_dst = torch.squeeze(membd[:, 2, :] + pe[:, 2, :], dim=1)'''

            h_i_src = torch.squeeze(membd[:, 0, :], dim=1)
            h_p = torch.squeeze(membd[:, 1, :], dim=1)
            h_i_dst = torch.squeeze(membd[:, 2, :], dim=1)

            h_i_src = self.linear_i(h_i_src)
            h_p = self.linear_p(h_p)
            h_i_dst = self.linear_i(h_i_dst)

            return h_i_src + h_p + h_i_dst

        if mtype == 1:
            '''max_seq_len = 4
            pe = self.positional_encoding(membd.shape[0], max_seq_len, self.hidden_dim)
            h_i_src = torch.squeeze(membd[:, 0, :] + pe[:, 0, :], dim=1)
            h_p1 = torch.squeeze(membd[:, 1, :] + pe[:, 1, :], dim=1)
            h_p2 = torch.squeeze(membd[:, 2, :] + pe[:, 2, :], dim=1)
            h_i_dst = torch.squeeze(membd[:, 3, :] + pe[:, 3, :], dim=1)'''

            h_i_src = torch.squeeze(membd[:, 0, :], dim=1)
            h_p1 = torch.squeeze(membd[:, 1, :], dim=1)
            h_p2 = torch.squeeze(membd[:, 2, :], dim=1)
            h_i_dst = torch.squeeze(membd[:, 3, :], dim=1)

            h_i_src = self.linear_i(h_i_src)
            h_p1 = self.linear_p(h_p1)
            h_p2 = self.linear_p(h_p2)
            h_i_dst = self.linear_i(h_i_dst)

            return h_i_src + h_p1 + h_p2 + h_i_dst

        if mtype == 2:
            '''max_seq_len = 5
            pe = self.positional_encoding(membd.shape[0], max_seq_len, self.hidden_dim)
            h_i_src = torch.squeeze(membd[:, 0, :] + pe[:, 0, :], dim=1)
            h_p1 = torch.squeeze(membd[:, 1, :] + pe[:, 1, :], dim=1)
            h_pa = torch.squeeze(membd[:, 2, :] + pe[:, 2, :], dim=1)
            h_p2 = torch.squeeze(membd[:, 3, :] + pe[:, 3, :], dim=1)
            h_i_dst = torch.squeeze(membd[:, 4, :] + pe[:, 4, :], dim=1)'''

            h_i_src = torch.squeeze(membd[:, 0, :], dim=1)
            h_p1 = torch.squeeze(membd[:, 1, :], dim=1)
            h_pa = torch.squeeze(membd[:, 2, :], dim=1)
            h_p2 = torch.squeeze(membd[:, 3, :], dim=1)
            h_i_dst = torch.squeeze(membd[:, 4, :], dim=1)

            h_i_src = self.linear_i(h_i_src)
            h_p1 = self.linear_p(h_p1)
            h_pa = self.linear_pa(h_pa)
            h_p2 = self.linear_p(h_p2)
            h_i_dst = self.linear_i(h_i_dst)

            return h_i_src + h_p1 + h_pa + h_p2 + h_i_dst

    def edge_softmax(self, g):
        attention = self.softmax(g, g.edata.pop('a'))
        # Dropout attention scores and save them
        g.edata['a_drop'] = self.dropout(attention) # aPvu

    def message_passing(self, edges):
        ft = edges.data['efeature'] * edges.data['a_drop']
        return {'ft': ft}

    def gated_multihead_attention(self, ret):
        gate = torch.mean(self.fc_gate(ret), dim=0)
        return gate.unsqueeze(dim=0)

    def forward(self, batch_nodes, metapath_indices, metapath_graph: dgl.DGLGraph, transformed_feature):
        metapath_graph.ndata['f'] = transformed_feature[:metapath_graph.num_nodes(), :]
        # node type-specific aggregation
        metapath_embd = F.embedding(metapath_indices, transformed_feature) # [num_indices, seq, hidden_dim]
        edata = self.node_type_specific_agg(self.mtype, metapath_embd) # [num_indices, hidden_dim]
        # edata = torch.mean(metapath_embd, dim=1, keepdim=False)

        # intra-metapath aggregation
        hidden = torch.cat([edata] * self.num_heads, dim=1) # [num_indices, num_heads * hidden_dim]
        hidden = torch.unsqueeze(hidden, dim=0) # [1, num_indices, num_heads * hidden_dim]
        efeature = hidden.permute(1, 0, 2).view(-1, self.num_heads, self.hidden_dim)  # [num_indices, num_heads, hidden_dim]
        # a = (efeature * self.attn).sum(dim=-1).unsqueeze(dim=-1)  # [num_indices, num_heads, 1]

        # talking-heads
        a = self.linear_talking_heads((efeature * self.attn).sum(dim=-1)).unsqueeze(dim=-1)  # [num_indices, num_logits_heads, 1]

        a = self.leaky_relu(a)
        metapath_graph.edata.update({'efeature': efeature, 'a': a})
        self.edge_softmax(metapath_graph)
        metapath_graph.update_all(self.message_passing, fn.sum('ft', 'ft'))
        '''if self.mtype == 0:
            result = metapath_graph.ndata['ft'][batch_nodes]
        if self.mtype == 1:
            result = metapath_graph.ndata['ft'][batch_nodes]'''
        result = metapath_graph.ndata['ft'][batch_nodes]
        # gated multi-head attention
        attn_gate = self.gated_multihead_attention(result)
        result = result * attn_gate # [batch_size, num_heads, hidden_dim]
        result = result.view(-1, self.num_heads * self.hidden_dim)

        # 孤立ipc节点的向量为transformed feature
        '''result_filter = torch.sum(result, dim=1).unsqueeze(dim=1)
        zero_index = (result_filter==0).nonzero()[:, 0]
        zero_feature = transformed_feature[batch_nodes[zero_index]]
        zero_feature = torch.cat(self.num_heads * [zero_feature], dim=1)
        result[zero_index] = result[zero_index] + zero_feature'''
        return result

class InterMetapathAggregation(nn.Module):
    def __init__(self, args):
        super(InterMetapathAggregation, self).__init__()
        self.fc1 = nn.Linear(in_features=args.num_heads * args.hidden_dim, out_features=args.hidden_dim, bias=False)
        self.fc2 = nn.Linear(in_features=args.hidden_dim, out_features=1, bias=False)

    def forward(self, intra_outs: torch.Tensor):
        beta = []
        for metapath_out in intra_outs:
            # metapath_out.shape = [batch_size, num_heads * hidden_dim]
            fc1 = torch.tanh(self.fc1(metapath_out))  # [batch_size, hidden_dim]
            fc1_mean = torch.mean(fc1, dim=0)
            fc2 = self.fc2(fc1_mean)
            beta.append(fc2)
        beta = torch.cat(beta, dim=0)
        beta = F.softmax(beta, dim=0)
        beta = torch.unsqueeze(beta, dim=-1)
        beta = torch.unsqueeze(beta, dim=-1)
        metapath_outs = [torch.unsqueeze(metapath_out, dim=0) for metapath_out in intra_outs]
        metapath_outs = torch.cat(metapath_outs, dim=0)
        result = torch.sum(beta * metapath_outs, dim=0)  # [batch_size, num_heads * hidden_dim]
        return result

class MTHG(nn.Module):
    def __init__(self, feature_dict, args):
        super(MTHG, self).__init__()
        self.args = args
        self.dropout = nn.Dropout(args.dropout_rate)
        self.feature_trasform = FeatureTransformLayer(feature_dict, args)
        self.intra_metapath_aggregation_layers = nn.ModuleList()
        for mtype in range(args.num_metapath_types):
            self.intra_metapath_aggregation_layers.append(IntraMetapathAggregation(args, mtype))
        self.inter_metapath_aggregation = InterMetapathAggregation(args)
        self.lstm = nn.LSTM(input_size=args.num_heads * args.hidden_dim, hidden_size=args.output_dim, batch_first=True, dropout=self.args.dropout_rate)

    def forward(self, inputs):
        batch, metapaths, metapath_graphs, timestamp, mode = inputs
        transformed_feature = self.dropout(self.feature_trasform())
        metapath_outputs = []
        for t in range(timestamp - self.args.min_timestamp + 1):

            # feature_t = transformed_feature[:, t, :]

            intra_metapath_outputs = []

            metapaths_t = metapaths[t]

            metapath_graph_t = metapath_graphs[t]

            for mtype in range(self.args.num_metapath_types):

                mtype_indices_t = metapaths_t[mtype]

                mtype_graph_t = metapath_graph_t[mtype]

                mtype_intra_output_t = self.intra_metapath_aggregation_layers[mtype](batch, mtype_indices_t, mtype_graph_t, transformed_feature)
                # mtype_intra_output_t = self.intra_metapath_aggregation_layers[mtype](batch, mtype_indices_t, mtype_graph_t, feature_t)

                if self.args.metapath_masks[mtype] != 0:
                    intra_metapath_outputs.append(mtype_intra_output_t)
                else:
                    intra_metapath_outputs.append(torch.randn_like(mtype_intra_output_t, dtype=torch.float32))

            metapath_output_t = self.inter_metapath_aggregation(intra_metapath_outputs)
            metapath_outputs.append(torch.unsqueeze(metapath_output_t, dim=1))

        metapath_outputs_u = torch.cat(metapath_outputs, dim=1) # [batch_size, t, num_heads * hidden_dim]

        output, (_, _) = self.lstm(metapath_outputs_u)
        if mode == 'train':
            return output # [batch_size, t, output_dim]
        if mode == 'eval':
            return output[:, -2, :].squeeze(dim=1) # [batch_size, output_dim]





