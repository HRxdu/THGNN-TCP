import dgl
import torch
import pickle
import pandas as pd

def edge_filter(edges, t):
    return edges.data['t'] != t

def get_ipc2patent2ipc(graph: dgl.DGLGraph, node, num_ipc, etypes=['assigned to', 'assigned with']) -> list:
    i_src = node
    i2p = graph.out_edges(i_src, etype=etypes[0])
    if i2p[0].shape[0] == 0:
        return [3 * [i_src]]
    else:
        i2p_df = pd.DataFrame(list(zip(i2p[0].numpy().tolist(), i2p[1].numpy().tolist())), columns=['ipc1', 'patent'])
        p = i2p[1]
        p2i = graph.out_edges(p, etype=etypes[1])
        p2i_df = pd.DataFrame(list(zip(p2i[0].numpy().tolist(), p2i[1].numpy().tolist())), columns=['patent', 'ipc2'])
        join_df = pd.merge(i2p_df, p2i_df, left_on='patent', right_on='patent', how='right')
        join_df['patent'] = join_df['patent'] + num_ipc
        join_df = join_df[join_df['ipc1'] != join_df['ipc2']]
    return join_df.to_numpy().tolist()

def get_ipc2patent2patent(graph: dgl.DGLGraph, node, num_ipc,
                              etypes=['assigned to', ('patent', 'cites', 'patent'), ('patent', 'cited by', 'patent')]) -> list:
    result = []
    i_src = node
    i2p = graph.out_edges(i_src, etype=etypes[0])
    if i2p[0].shape[0] == 0:
        return [3 * [i_src]]
    else:
        i2p_df = pd.DataFrame(list(zip(i2p[0].numpy().tolist(), i2p[1].numpy().tolist())), columns=['ipc', 'patent1'])
        p_source = i2p[1]
        p2p_cites = graph.out_edges(p_source, etype=etypes[1])
        p2p_cited = graph.out_edges(p_source, etype=etypes[2])
        if p2p_cites[0].shape[0] == 0 and p2p_cited[0].shape[0] == 0:
            p_source = p_source.numpy().tolist()
            result_tmp = []
            for p in p_source:
                result_tmp.append([i_src, p, p])
            return result_tmp
        else:
            p2p_cites_df = pd.DataFrame(list(zip(p2p_cites[0].numpy().tolist(), p2p_cites[1].numpy().tolist())), columns=['patent1', 'patent2'])
            p2p_cited_df = pd.DataFrame(list(zip(p2p_cited[0].numpy().tolist(), p2p_cited[1].numpy().tolist())), columns=['patent1', 'patent2'])
            p2p_df = pd.concat([p2p_cites_df, p2p_cited_df])
            i2p2p_df = pd.merge(i2p_df, p2p_df, left_on='patent1', right_on='patent1', how='right').drop_duplicates()
            '''p_cited = p2p[1]
            p2i = graph.out_edges(p_cited, etype=etypes[2])
            p2i_df = pd.DataFrame(list(zip(p2i[0].numpy().tolist(), p2i[1].numpy().tolist())), columns=['patent2', 'ipc2'])
            i2p2p2i_df = pd.merge(i2p2p_df, p2i_df, left_on='patent2', right_on='patent2', how='right')'''
            i2p2p_df['patent1'] = i2p2p_df['patent1'] + num_ipc
            i2p2p_df['patent2'] = i2p2p_df['patent2'] + num_ipc
            return i2p2p_df.to_numpy().tolist()

def get_ipc2patent2patent2ipc(graph: dgl.DGLGraph, node, num_ipc,
                              etypes=['assigned to', ('patent', 'cites', 'patent'), ('patent', 'cited by', 'patent'), 'assigned with']) -> list:
    i_src = node
    i2p = graph.out_edges(i_src, etype=etypes[0])
    if i2p[0].shape[0] == 0:
        return [4 * [i_src]]
    else:
        i2p_df = pd.DataFrame(list(zip(i2p[0].numpy().tolist(), i2p[1].numpy().tolist())), columns=['ipc1', 'patent1'])
        p_source = i2p[1]
        p2p_cites = graph.out_edges(p_source, etype=etypes[1])
        p2p_cited = graph.out_edges(p_source, etype=etypes[2])
        if p2p_cites[0].shape[0] == 0 and p2p_cited[0].shape[0] == 0:
            p_source = p_source.numpy().tolist()
            result_tmp = []
            for p in p_source:
                result_tmp.append([i_src, p+num_ipc, p+num_ipc, i_src])
            return result_tmp
        else:
            p2p_cites_df = pd.DataFrame(list(zip(p2p_cites[0].numpy().tolist(), p2p_cites[1].numpy().tolist())), columns=['patent1', 'patent2'])
            p2p_cited_df = pd.DataFrame(list(zip(p2p_cited[0].numpy().tolist(), p2p_cited[1].numpy().tolist())), columns=['patent1', 'patent2'])
            p2p_df = pd.concat([p2p_cites_df, p2p_cited_df])
            i2p2p_df = pd.merge(i2p_df, p2p_df, left_on='patent1', right_on='patent1', how='right').drop_duplicates()

            p_cited = torch.tensor(list(i2p2p_df['patent2']))
            p2i = graph.out_edges(p_cited, etype=etypes[2])
            if p2i[0].shape[0] == 0:
                p_source = i2p[1].numpy().tolist()
                result_tmp = []

                for _, row in i2p2p_df.iterrows():
                    result_tmp.append([row['ipc1'], row['patent1']+num_ipc, row['patent2']+num_ipc, row['ipc1']])

                return result_tmp
            else:
                p2i_df = pd.DataFrame(list(zip(p2i[0].numpy().tolist(), p2i[1].numpy().tolist())), columns=['patent2', 'ipc2'])
                i2p2p2i_df = pd.merge(i2p2p_df, p2i_df, left_on='patent2', right_on='patent2', how='right').drop_duplicates()
                i2p2p2i_df['patent1'] = i2p2p2i_df['patent1'] + num_ipc
                i2p2p2i_df['patent2'] = i2p2p2i_df['patent2'] + num_ipc
                i2p2p2i_df = i2p2p2i_df[i2p2p2i_df['ipc1'] != i2p2p2i_df['ipc2']]
            return i2p2p2i_df.to_numpy().tolist()

def get_ipc2patent2paper(graph: dgl.DGLGraph, node, num_ipc, num_patent,
                                    etypes=['assigned to', ('patent', 'cites', 'paper')]) -> list:
    i_src = node
    i2patent = graph.out_edges(i_src, etype=etypes[0])
    if i2patent[0].shape[0] == 0:
        return [3 * [i_src]]
    else:
        i2patent_df = pd.DataFrame(list(zip(i2patent[0].numpy().tolist(), i2patent[1].numpy().tolist())), columns=['ipc', 'patent'])
        patent1 = i2patent[1]
        patent2paper = graph.out_edges(patent1, etype=etypes[1])
        if patent2paper[0].shape[0] == 0:
            return [3 * [i_src]]
        else:
            patent2paper_df = pd.DataFrame(list(zip(patent2paper[0].numpy().tolist(), patent2paper[1].numpy().tolist())), columns=['patent', 'paper'])
            i2patent2paper_df = pd.merge(i2patent_df, patent2paper_df, left_on='patent', right_on='patent', how='right').drop_duplicates()
            '''paper = patent2paper[1]
            paper2patent = graph.out_edges(paper, etype=etypes[2])
            paper2patent_df = pd.DataFrame(list(zip(paper2patent[0].numpy().tolist(), paper2patent[1].numpy().tolist())), columns=['paper', 'patent2'])
            i2patent2paper2patent_df = pd.merge(i2patent2paper_df, paper2patent_df, left_on='paper', right_on='paper', how='right')
            patent2 = paper2patent[1]
            patent2i = graph.out_edges(patent2, etype=etypes[3])
            patent2i_df = pd.DataFrame(list(zip(patent2i[0].numpy().tolist(), patent2i[1].numpy().tolist())), columns=['patent2', 'ipc2'])
            i2patent2paper2patent2i_df = pd.merge(i2patent2paper2patent_df, patent2i_df, left_on='patent2', right_on='patent2', how='right')
            i2patent2paper2patent2i_df['patent1'] = i2patent2paper2patent2i_df['patent1'] + num_ipc
            i2patent2paper2patent2i_df['patent2'] = i2patent2paper2patent2i_df['patent2'] + num_ipc'''
            i2patent2paper_df['patent'] = i2patent2paper_df['patent'] + num_ipc
            i2patent2paper_df['paper'] = i2patent2paper_df['paper'] + num_ipc + num_patent
        return i2patent2paper_df.to_numpy().tolist()

def get_i2patent2paper2patent2i(graph: dgl.DGLGraph, node, num_ipc, num_patent,
                                    etypes=['assigned to', ('patent', 'cites', 'paper'), ('paper', 'cited by', 'patent'), 'assigned with']) -> list:
    result = []
    i_src = node
    i2patent = graph.out_edges(i_src, etype=etypes[0])
    if i2patent[0].shape[0] == 0:
        return [5 * [i_src]]
    else:
        i2patent_df = pd.DataFrame(list(zip(i2patent[0].numpy().tolist(), i2patent[1].numpy().tolist())), columns=['ipc1', 'patent1'])
        patent1 = i2patent[1]
        patent2paper = graph.out_edges(patent1, etype=etypes[1])
        if patent2paper[0].shape[0] == 0:
            return [5 * [i_src]]
        else:
            patent2paper_df = pd.DataFrame(list(zip(patent2paper[0].numpy().tolist(), patent2paper[1].numpy().tolist())), columns=['patent1', 'paper'])
            i2patent2paper_df = pd.merge(i2patent_df, patent2paper_df, left_on='patent1', right_on='patent1', how='right').drop_duplicates()
            paper = patent2paper[1]
            paper2patent = graph.out_edges(paper, etype=etypes[2])
            if paper2patent[0].shape[0] == 0:
                return [5 * [i_src]]
            else:
                paper2patent_df = pd.DataFrame(list(zip(paper2patent[0].numpy().tolist(), paper2patent[1].numpy().tolist())), columns=['paper', 'patent2'])
                i2patent2paper2patent_df = pd.merge(i2patent2paper_df, paper2patent_df, left_on='paper', right_on='paper', how='right').drop_duplicates()
                patent2 = paper2patent[1]
                patent2i = graph.out_edges(patent2, etype=etypes[3])
                if patent2i[0].shape[0] == 0:
                    return [5 * [i_src]]
                else:
                    patent2i_df = pd.DataFrame(list(zip(patent2i[0].numpy().tolist(), patent2i[1].numpy().tolist())), columns=['patent2', 'ipc2'])
                    i2patent2paper2patent2i_df = pd.merge(i2patent2paper2patent_df, patent2i_df, left_on='patent2', right_on='patent2', how='right').drop_duplicates()
                    i2patent2paper2patent2i_df['patent1'] = i2patent2paper2patent2i_df['patent1'] + num_ipc
                    i2patent2paper2patent2i_df['patent2'] = i2patent2paper2patent2i_df['patent2'] + num_ipc
                    i2patent2paper2patent2i_df['paper'] = i2patent2paper2patent2i_df['paper'] + num_ipc + num_patent
                    i2patent2paper2patent2i_df = i2patent2paper2patent2i_df[i2patent2paper2patent2i_df['ipc1'] != i2patent2paper2patent2i_df['ipc2']]
        return i2patent2paper2patent2i_df.to_numpy().tolist()

def get_metapaths(graph_path, node, t, num_ipc, num_patent):
    '''
    因node_feature.shape=[num_ipc + num_patent + num_paper, feature_dim]：
    为使模型中方便使用torch.nn.F.embedding，
    ipc索引从0开始
    patent索引从num_ipc开始
    paper索引从num_ipc + num_patent开始

    :param graph_path:
    :param node:
    :param timestamp:
    :return: {
        node:
        {
            t: [i2patent2i, i2patent2patent, i2patent2paper]
            (
            i2patent2i.shape = [num_i2patent2i_indices, seq1]
            i2patent2patent.shape = [num_i2patent2patent_indices, seq2]
            i2patent2paper.shape = [num_i2patent2paper_indices, seq3]
            )
        }
    }
    '''
    graph_t = dgl.load_graphs(graph_path)[0][0]
    for etype in graph_t.canonical_etypes:
        filtered_edges = graph_t.filter_edges(lambda x: edge_filter(x, t=t), etype=etype)
        graph_t.remove_edges(eids=filtered_edges, etype=etype)

    ipc2patent2ipc = torch.tensor(get_ipc2patent2ipc(graph_t, node, num_ipc))
    ipc2patent2patent2ipc = torch.tensor(get_ipc2patent2patent2ipc(graph_t, node, num_ipc))
    # ipc2patent2patent = torch.tensor(get_ipc2patent2patent(graph_t, node, num_ipc))
    # ipc2patent2paper = torch.tensor(get_ipc2patent2paper(graph_t, node, num_ipc, num_patent))
    i2patent2paper2patent2i = torch.tensor(get_i2patent2paper2patent2i(graph_t, node, num_ipc, num_patent))

    return [ipc2patent2ipc, ipc2patent2patent2ipc, i2patent2paper2patent2i]

if __name__ == '__main__':
    graph_path = '../result/大模型/thg.dgl'
    result_path = '../result/大模型/'
    graph = dgl.load_graphs(graph_path)[0][0]
    nodes = graph.nodes(ntype='ipc')
    num_ipc = nodes.shape[0]
    num_patent = graph.num_nodes(ntype='patent')
    result = {}
    for node in nodes:
        print(node)
        node = int(node.numpy())
        result[node] = {}
        for t in range(2010, 2024):
            result[node][t] = get_metapaths(graph_path, node, t, num_ipc, num_patent)

    with open(result_path + 'temporal_metapath_dict_paper1.pkl', 'wb') as f:
        pickle.dump(result, f)
    '''result_path = '../result/大模型/'
    with open(result_path + 'temporal_metapath_dict.pkl', 'rb') as f:
        test = pickle.load(f)
    print(test)'''