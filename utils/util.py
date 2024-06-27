import dgl
import torch
import pickle

import numpy as np
import networkx as nx
import scipy.sparse as sp

from arguments import args
from dgl.dataloading.negative_sampler import GlobalUniform

with open(args.metapath_path, 'rb') as f:
    metapath_dict = pickle.load(f)
    f.close()

def edge_filter4now(edges, t):
    return edges.data['t'] == t

def edge_filter4not_now(edges, t):
    return edges.data['t'] != t

def get_train_graph(graph_path=args.graph_path, timestamp=args.test_timestamp):
    graph = dgl.load_graphs(graph_path)[0][0]
    graph.remove_edges(graph.filter_edges(lambda x: edge_filter4now(x, timestamp), etype='co-occurs with'), etype='co-occurs with')
    result = dgl.edge_type_subgraph(graph, etypes=['co-occurs with'])
    return result

def get_test_graph(graph_path=args.graph_path, timestamp=args.test_timestamp):
    graph = dgl.load_graphs(graph_path)[0][0]
    graph.remove_edges(graph.filter_edges(lambda x: edge_filter4not_now(x, timestamp), etype='co-occurs with'), etype='co-occurs with')
    result = dgl.edge_type_subgraph(graph, etypes=['co-occurs with'])
    return result

def get_node_pairs(graphs, nodes: torch.Tensor):
    neg_sampler = GlobalUniform(k=2)
    all_nodes = nodes.cpu().numpy().tolist()
    result = {}
    for tidx, t in enumerate(args.time_range):
        result_t = {}
        graph_t = graphs[tidx]
        graph_t = dgl.add_self_loop(graph_t)
        pos_edges = graph_t.out_edges(nodes)
        pos_nodes = pos_edges[1].numpy().tolist()
        num_pos = len(pos_nodes)

        neg_edges = neg_sampler(graph_t, eids=graph_t.edge_ids(u=pos_edges[0], v=pos_edges[1]))
        neg_edges = tuple(map(lambda x: x[:num_pos], neg_edges))
        neg_nodes = neg_edges[0].numpy().tolist() + neg_edges[1].numpy().tolist()
        all_nodes += pos_nodes + neg_nodes
        result_t['pos'] = pos_edges
        result_t['neg'] = neg_edges
        result[t] = result_t
    all_nodes = list(set(all_nodes))
    all_nodes = torch.tensor(all_nodes, dtype=torch.long, device=args.device)
    return all_nodes, result

def get_metapaths_till_t(nodes: torch.Tensor, timestamp: int, metapath_dict: dict=metapath_dict, args=args):
    '''

    :param nodes:
    :param timestamp:
    :param metapath_dict:
    :return: num_timestamps * [num_mtypes * [ Tensor.shape=[num_indices, mtype_len] ] ]
    '''
    result = []
    for _ in range(args.min_timestamp, timestamp+1):
        result_tmp = []
        for mtype in range(args.num_metapath_types):
            result_mtype = []
            for node in nodes:
                node = int(node.cpu().numpy())
                metapaths_t = metapath_dict[node][timestamp]
                metapaths_t_mtype = metapaths_t[mtype].numpy().tolist()
                result_mtype += metapaths_t_mtype
            result_tmp.append(torch.tensor(result_mtype, device=args.device))
        result.append(result_tmp)
    return result

def get_metapath_graphs_till_t(metapaths, timestamp: int, args=args):
    result = []
    '''metapath_dict = {
        'similar patent-cites': [('ipc', 'assigned to', 'patent'), ('patent', 'cites', 'patent')],
        'similar patent-cited': [('ipc', 'assigned to', 'patent'), ('patent', 'cited by', 'patent')],
        'similar paper': [('ipc', 'assigned to', 'patent'), ('patent', 'cites', 'paper')]
    }'''
    # transform = dgl.AddMetaPaths(metapath_dict, keep_orig_edges=True)
    # new_graph = transform(graph)
    for t in range(timestamp - args.min_timestamp + 1):
        result_tmp = []
        metapath_indices_t = metapaths[t]
        for mtype in range(args.num_metapath_types):
            metapath_indices_t_mtype = metapath_indices_t[mtype]
            if mtype == 0:
                # i2patent2i
                ipc_src = metapath_indices_t_mtype[:, 0]
                ipc_dst = metapath_indices_t_mtype[:, 2]
                graph_tmp = dgl.DGLGraph().to(args.device)
                graph_tmp.add_nodes(args.num_ipc)
                graph_tmp.add_edges(u=ipc_dst, v=ipc_src)
                # ipcs = list(set(torch.cat([ipc_src, ipc_dst]).cpu().numpy().tolist()))
                # graph_tmp = dgl.node_subgraph(new_graph, nodes={'ipc': torch.tensor(ipcs)})
                # graph_tmp = dgl.edge_type_subgraph(graph_tmp, etypes=['co-occur']).to(args.device)
                result_tmp.append(graph_tmp)

            '''if mtype == 1:
                # i2patent2patent
                src = metapath_indices_t_mtype[:, 0]
                dst = metapath_indices_t_mtype[:, 2]
                graph_tmp = dgl.DGLGraph().to(args.device)
                graph_tmp.add_nodes(args.num_ipc + args.num_patent)
                graph_tmp.add_edges(u=dst, v=src)
                result_tmp.append(graph_tmp)'''

            if mtype == 1:
                # i2patent2patent2i
                src = metapath_indices_t_mtype[:, 0]
                dst = metapath_indices_t_mtype[:, 3]
                graph_tmp = dgl.DGLGraph().to(args.device)
                graph_tmp.add_nodes(args.num_ipc + args.num_patent)
                graph_tmp.add_edges(u=dst, v=src)
                result_tmp.append(graph_tmp)

            '''if mtype == 2:
                # i2patent2paper
                src = metapath_indices_t_mtype[:, 0]
                dst = metapath_indices_t_mtype[:, 2]
                graph_tmp = dgl.heterograph({
                    ('paper', 'to', 'ipc'): (dst, src)
                }).to(args.device)
                result_tmp.append(graph_tmp)'''

            if mtype == 2:
                # i2patent2paper2patent2ipc
                src = metapath_indices_t_mtype[:, 0]
                dst = metapath_indices_t_mtype[:, 4]
                graph_tmp = dgl.DGLGraph().to(args.device)
                graph_tmp.add_nodes(args.num_ipc + args.num_patent + args.num_paper)
                graph_tmp.add_edges(u=dst, v=src)
                result_tmp.append(graph_tmp)
        result.append(result_tmp)
    return result

def get_evaluation_data(train_adj, test_adj):
    """ Load val/test examples to evaluate link prediction performance"""
    val_edges, val_edges_false, test_edges, test_edges_false = \
        create_data_splits(train_adj, test_adj, val_mask_fraction=0.5, test_mask_fraction=0.5)

    return val_edges, val_edges_false, test_edges, test_edges_false

def create_data_splits(adj, next_adj, val_mask_fraction, test_mask_fraction):
    """In: (adj, next_adj) along with test and val fractions. For link prediction (on all links), all links in
    next_adj are considered positive examples.
    Out: list of positive and negative pairs for link prediction (train/val/test)"""
    edges_all = sparse_to_tuple(next_adj)[0]  # All edges in original adj.
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)  # Remove diagonal elements
    adj.eliminate_zeros()
    assert np.diag(adj.todense()).sum() == 0
    if next_adj is None:
        raise ValueError('Next adjacency matrix is None')

    edges_next = np.array(list(set(nx.from_scipy_sparse_matrix(next_adj).edges())))
    edges = []   # Constraint to restrict new links to existing nodes.
    for e in edges_next:
        if e[0] < adj.shape[0] and e[1] < adj.shape[0]:
            edges.append(e)
    edges = np.array(edges)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    num_test = int(np.floor(edges.shape[0] * test_mask_fraction))
    num_val = int(np.floor(edges.shape[0] * val_mask_fraction))
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    # train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    # Create train edges.
    '''train_edges_false = []
    while len(train_edges_false) < len(train_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
            continue
        if train_edges_false:
            if ismember([idx_j, idx_i], np.array(train_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(train_edges_false)):
                continue
        train_edges_false.append([idx_i, idx_j])'''

    # Create test edges.
    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    # Create val edges.
    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
            continue

        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    # assert ~ismember(val_edges, train_edges)
    # assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)
    # print("# train examples: ", len(train_edges), len(train_edges_false))
    print("# val examples:", len(val_edges), len(val_edges_false))
    print("# test examples:", len(test_edges), len(test_edges_false))

    return torch.tensor(list(val_edges), dtype=torch.long), torch.tensor(val_edges_false, dtype=torch.long), torch.tensor(list(test_edges), dtype=torch.long), torch.tensor(test_edges_false, dtype=torch.long)

def sparse_to_tuple(sparse_mx):
    """Convert scipy sparse matrix to tuple representation (for tf feed dict)."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    def to_tuple_list(matrices):
        # Input is a list of matrices.
        coords = []
        values = []
        shape = [len(matrices)]
        for i in range(0, len(matrices)):
            mx = matrices[i]
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            # Create proper indices - coords is a numpy array of pairs of indices.
            coords_mx = np.vstack((mx.row, mx.col)).transpose()
            z = np.array([np.ones(coords_mx.shape[0]) * i]).T
            z = np.concatenate((z, coords_mx), axis=1)
            z = z.astype(int)
            coords.extend(z)
            values.extend(mx.data)

        shape.extend(matrices[0].shape)
        shape = np.array(shape).astype("int64")
        values = np.array(values).astype("float32")
        coords = np.array(coords)
        return coords, values, shape

    if isinstance(sparse_mx, list) and isinstance(sparse_mx[0], list):
        # Given a list of lists, convert it into a list of tuples.
        for i in range(0, len(sparse_mx)):
            sparse_mx[i] = to_tuple_list(sparse_mx[i])

    elif isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def write_to_csv(test_result, output_name, dataset, mod='test'):
    """Output result scores to a csv file for result logging"""
    with open(output_name, 'a+') as f:
        print("test result ({})".format(mod), test_result[0], test_result[1])
        # best_auc = test_result
        f.write("{},{},{},{}\n".format(dataset,  mod, "AUC", test_result[0]))
        f.write("{},{},{},{}\n".format(dataset,  mod, "F1", test_result[1]))