import os
import json
import pickle
import dgl
import time

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import logging

from scipy.sparse import csr_matrix
from process.ipc_adj_matrix import ipc_transform
from myModel import MTHG
from arguments import args
from datetime import datetime
from utils.pytorchtools import EarlyStopping
from utils.dataset import LLMDataset_node, LLMDataset_eval, LLMDataset_t, LLMDataset_edge
from utils.util import *
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

if __name__ == '__main__':
    # Assumes a saved base model as input and model name to get the right directory.
    data_dir = 'result/{}/'.format(args.dataset)
    output_dir = data_dir + 'model_output/'.format(args.dataset)


    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Set paths of sub-directories.
    LOG_DIR = output_dir + args.log_dir
    SAVE_DIR = output_dir + args.save_dir
    CSV_DIR = output_dir + args.csv_dir
    MODEL_DIR = output_dir + args.model_dir

    if not os.path.isdir(LOG_DIR):
        os.mkdir(LOG_DIR)

    if not os.path.isdir(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    if not os.path.isdir(CSV_DIR):
        os.mkdir(CSV_DIR)

    if not os.path.isdir(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID

    # Load graphs and features.
    device = args.device
    graph = dgl.load_graphs(args.graph_path)[0][0]
    print(graph)
    edges = graph.edges(etype='co-occurs with')
    edges = set(zip(edges[0].numpy().tolist(), edges[1].numpy().tolist()))
    graphs = dgl.load_graphs(args.graphs_path)[0]
    feature_dict = graph.ndata['f'] # keys(['ipc', 'paper', 'patent'])

    mode = 'eval'
    model = MTHG(feature_dict, args).to(device)
    model.load_state_dict(torch.load(output_dir + 'model/checkpoint.pt'))

    data = pd.read_excel(data_dir + '2023/gephi_data.xlsx', sheet_name='node')
    ipc2id = json.load(open(data_dir + 'ipc2id.json', 'r'))

    target_ipc = list(data['Id'])
    target_ipc = list(map(ipc_transform, target_ipc))
    target_ipc = torch.tensor(list(map(lambda x: ipc2id[x], target_ipc)))

    target_ipc_list = target_ipc.numpy().tolist()
    idx2ipc = dict(zip(list(range(len(target_ipc_list))), target_ipc_list))

    metapaths_till_t = get_metapaths_till_t(target_ipc, timestamp=args.test_timestamp)
    metapath_graphs_till_t = get_metapath_graphs_till_t(metapaths_till_t, timestamp=args.test_timestamp)

    embd = model([target_ipc, metapaths_till_t, metapath_graphs_till_t, args.test_timestamp, mode])

    embd_u = embd.view(-1, embd.shape[-1])
    embd_v = embd.view(embd.shape[-1], -1)

    prob_adj = torch.sigmoid(torch.mm(embd_u, embd_v)).detach().cpu().numpy()

    # 阈值筛选
    p = 0.9
    topn = 5

    # 找到所有大于p的元素的位置和对应的值
    values_above_threshold = prob_adj[prob_adj > p]
    indices = np.nonzero(prob_adj > p)

    # 将索引与对应的值配对，并按值进行倒序排序
    indexed_values = np.vstack((indices[0], indices[1], values_above_threshold)).T
    sorted_indices_values = indexed_values[indexed_values[:, 2].argsort()[::-1]]

    # 获取排序后的横纵坐标
    sorted_row_indices = sorted_indices_values[:, 0]
    sorted_col_indices = sorted_indices_values[:, 1]

    for index, (row, col) in enumerate(zip(sorted_row_indices, sorted_col_indices)):
        if index < topn:
            row = int(row)
            col = int(col)
            print(f"({idx2ipc[row]}, {idx2ipc[col]}) {prob_adj[row][col]}")

    map_func = np.frompyfunc(lambda x: 1 if x >= p else 0, 1, 1)
    prediction_result = map_func(prob_adj)

    row, col = np.diag_indices_from(prediction_result)
    prediction_result[row, col] = 0

    prediction_result = np.triu(prediction_result, k=0)
    prediction_result = csr_matrix(prediction_result, dtype=int)

    prediction_result = list(zip(*prediction_result.nonzero()))
    prediction_result = set(map(lambda x: (idx2ipc[x[0]], idx2ipc[x[1]]), prediction_result))

    new_edges = prediction_result.difference(edges)

    print(new_edges)
    print(len(new_edges))

    '''with open(f'result/大模型/prediction_analysis_p={p}.pkl', 'wb') as f:
        pickle.dump(prediction_result, f)'''









