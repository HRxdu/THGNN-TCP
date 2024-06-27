import dgl
import torch
import networkx as nx
import numpy as np
import pandas as pd
# import easygraph as eg
from arguments import args
from utils.util import get_train_graph, get_test_graph, get_evaluation_data
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
# from easygraph.functions.graph_embedding.deepwalk import deepwalk
# from model.line.line import LINE

def cosine_similarity(x, y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom

if __name__ == '__main__':
    result_path = 'result/大模型/'
    dgl_g = dgl.load_graphs(args.graph_path)[0][0]
    dgl_g = dgl.edge_type_subgraph(dgl_g, etypes=['co-occurs with'])
    adj = dgl_g.adjacency_matrix(scipy_fmt='csr')
    ug = nx.from_scipy_sparse_matrix(adj)

    train_graph = get_train_graph()
    test_graph = get_test_graph()
    train_adj = train_graph.adjacency_matrix(scipy_fmt='csr')
    test_adj = test_graph.adjacency_matrix(scipy_fmt='csr')
    _, _, test_edges, test_edges_false = get_evaluation_data(train_adj, test_adj)
    test_edges = test_edges.numpy().tolist()
    test_edges_false = test_edges_false.numpy().tolist()

    edges_test = test_edges + test_edges_false
    labels_test = [1] * len(test_edges) + [0] * len(test_edges_false)
    result = {'labels': labels_test}

    th = 0.5
    # 2.1.2 PA方法
    edges_test_int = [(int(u), int(v)) for u, v in edges_test]
    print('2.1.2 PA方法')
    pa_prob = [p for u, v, p in nx.preferential_attachment(ug, edges_test_int)]
    result['PA_prob'] = minmax_scale(pa_prob).tolist()
    del pa_prob
    result['PA_label'] = [1 if x >= th else 0 for x in result['PA_prob']]
    # 2.1.3 JC方法
    print('2.1.3 JC方法')
    jc_prob = [p for u, v, p in nx.jaccard_coefficient(ug, edges_test_int)]
    result['JC_prob'] = minmax_scale(jc_prob).tolist()
    del jc_prob
    result['JC_label'] = [1 if x >= th else 0 for x in result['JC_prob']]
    # 2.1.4 AA方法
    print('2.1.4 AA方法')
    aa_prob = [p for u, v, p in nx.adamic_adar_index(ug, edges_test_int)]
    result['AA_prob'] = minmax_scale(aa_prob).tolist()
    del aa_prob
    result['AA_label'] = [1 if x >= th else 0 for x in result['AA_prob']]
    # 2.1.5 DW方法
    '''print('2.1.5 DW方法')

    skip_gram_params = dict(  # The skip_gram parameters in Python package gensim.
        window=10,
        min_count=1)
    embeddings_dw, _ = deepwalk(egg, dimensions=128, walk_length=80, num_walks=10, **skip_gram_params)
    dw_prob = [cosine_similarity(embeddings_dw[int(src)], embeddings_dw[int(dst)]) for src, dst in edges_test]
    result['DW_prob'] = minmax_scale(dw_prob).tolist()
    del dw_prob
    result['DW_label'] = [1 if x >= th else 0 for x in result['DW_prob']]'''
    # 2.1.6 LINE方法
    '''print('2.1.6 LINE方法')
    model_line = LINE(ug, embedding_size=128, order='second')  # init model,order can be ['first','second','all']
    model_line.train(batch_size=1024, epochs=50, verbose=2)  # train model
    embeddings_line = model_line.get_embeddings()  # get embedding vectors
    line_prob = [cosine_similarity(embeddings_line[int(src)], embeddings_line[int(dst)])
                 for src, dst in edges_test]
    result['LINE_prob'] = minmax_scale(line_prob).tolist()
    del line_prob
    result['LINE_label'] = [1 if x >= th else 0 for x in result['LINE_prob']]'''
    # 2.1.7 SDNE方法
    '''print('2.1.7 SDNE方法')
    model_sdne = SDNE(egg, hidden_size=[256, 128])  # init model
    model_sdne.train(batch_size=1874, epochs=40, verbose=2)  # train model
    embeddings_sdne = model_sdne.get_embeddings()  # get embedding vectors
    sdne_prob = [cosine_similarity(embeddings_sdne[int(src)], embeddings_sdne[int(dst)])
                 for src, dst in edges_test]
    result['SDNE_prob'] = minmax_scale(sdne_prob).tolist()
    del sdne_prob
    result['SDNE_label'] = [1 if x >= th else 0 for x in result['SDNE_prob']]'''

    pd.DataFrame.from_dict(result).to_csv(result_path + '/static_result.csv')

    # 2.2 计算指标
    print('各类方法均以{}为阈值计算各项指标'.format(th))
    for method in ['PA', 'JC', 'AA']:
        print('{}方法指标：'.format(method))
        print('混淆矩阵：\n', confusion_matrix(result['labels'], result['{}_label'.format(method)]))
        print('AUC值为：\n', roc_auc_score(result['labels'], result['{}_prob'.format(method)]))
        print('ACC:\n', accuracy_score(result['labels'], result['{}_label'.format(method)]))
        print('PRE:\n', precision_score(result['labels'], result['{}_label'.format(method)]))
        print('REC:\n', recall_score(result['labels'], result['{}_label'.format(method)]))
        print('F1:\n', f1_score(result['labels'], result['{}_label'.format(method)]))