import sys

import dgl
import json
import torch
import pandas as pd

from ipc_adj_matrix import ipc_transform
from itertools import chain

def get_patent_patent_data(data_path: str, patent2id: dict):
    print('get patent2patent data')
    data1 = pd.read_excel(data_path + '1-10000.xlsx', usecols=['公开（公告）号', '引证专利', 'IPC']).rename(columns={'公开（公告）号': 'patent', '引证专利': 'inf_patent'})
    data2 = pd.read_excel(data_path + '10001-13517.xlsx', usecols=['公开（公告）号', '引证专利', 'IPC']).rename(columns={'公开（公告）号': 'patent', '引证专利': 'inf_patent'})
    data = pd.concat([data1, data2]).dropna(subset=['IPC'])
    data['patent'] = data['patent'].apply(lambda x: patent2id[x])

    # 添加自环
    source = list(data['patent'])
    target = list(data['patent'])

    data = data.dropna(subset=['patent', 'inf_patent'], how='any')

    def inf_filter(inf_patents):
        inf_set = set(inf_patents.split('; '))
        patent_set = set(list(patent2id.keys()))

        if len(inf_set.intersection(patent_set)):
            return list(inf_set.intersection(patent_set))
        else:
            return None

    data['inf_patent'] = data['inf_patent'].apply(inf_filter)
    data = data.dropna(subset=['patent', 'inf_patent'], how='any')
    data['inf_patent'] = data['inf_patent'].apply(lambda x: list(map(lambda p: patent2id[p], x)))

    patents = list(data['patent'])
    inf_patents = list(data['inf_patent'])
    patent2patent = dict(zip(patents, inf_patents))

    for patent, inf_patents in patent2patent.items():
        source += len(inf_patents) * [patent]
        target += inf_patents

    assert len(source) == len(target)
    print('finish')

    return torch.tensor(source), torch.tensor(target)

def get_patent_paper_data(data_path: str, patent2id: dict, paper2id: dict):
    print('get patent2paper data')
    papers = pd.read_excel(data_path + 'papers.xlsx')

    source = []
    target = []

    papers['Matched Publication Numbers'] = papers['Matched Publication Numbers'].apply(lambda x: list(map(lambda patent: patent2id[patent], x.split(', '))))

    patents = list(papers['Matched Publication Numbers'])
    papers = list(papers['Title'].apply(lambda x: paper2id[x]))

    for index, i in enumerate(patents):
        source += i
        target += len(i) * [papers[index]]

    assert len(source) == len(target)
    print('finish')

    return torch.tensor(source), torch.tensor(target)

def get_patent_ipc_data(data_path: str, ipc2id: dict, patent2id:dict):
    print('get patent2ipc data')
    data = pd.read_excel(data_path + 'data.xlsx')
    source = []
    target = []

    data['IPC'] = data['IPC'].apply(lambda x: list(map(lambda ipc: ipc2id[ipc_transform(ipc)], x.split('; '))))
    patent2ipc = dict(zip(list(data['公开（公告）号']), list(data['IPC'])))

    for patent, ipcs in patent2ipc.items():
        patent_id = patent2id[patent]
        source += len(ipcs) * [patent_id]
        target += ipcs

    assert len(source) == len(target)
    print('finish')

    return torch.tensor(source), torch.tensor(target)

def get_cooccur_data(data_path: str, ipc2id: dict):
    print('get co-occur data')
    source = []
    target = []
    for year in range(2010, 2024):
        data_tmp = pd.read_excel(data_path + f'{year}/gephi_data.xlsx', sheet_name='edge')
        source_tmp = list(data_tmp['Source'].apply(ipc_transform).apply(lambda x: ipc2id[x]))
        target_tmp = list(data_tmp['Target'].apply(ipc_transform).apply(lambda x: ipc2id[x]))
        source += source_tmp
        target += target_tmp

    assert len(source) == len(target)
    print('finish')

    return torch.tensor(source), torch.tensor(target)

def prepare_graph_data(processed_data_path: str, ori_data_path: str, ipc2id: dict, patent2id:dict, paper2id: dict) -> dict:
    print('1.prepare graph data:')
    cooccur_source, cooccur_target = get_cooccur_data(processed_data_path, ipc2id)
    patent2ipc_source, patent2ipc_target = get_patent_ipc_data(ori_data_path, ipc2id, patent2id)
    patent2paper_source, patent2paper_target = get_patent_paper_data(ori_data_path, patent2id, paper2id)
    patent2patent_source, patent2patent_target = get_patent_patent_data(ori_data_path, patent2id)

    print('1.graph data prepared')

    return {
        ('ipc', 'co-occurs with', 'ipc'): (cooccur_source, cooccur_target),
        ('ipc', 'assigned to', 'patent'): (patent2ipc_target, patent2ipc_source),
        ('patent', 'assigned with', 'ipc'): (patent2ipc_source, patent2ipc_target),
        ('patent', 'cites', 'paper'): (patent2paper_source, patent2paper_target),
        ('paper', 'cited by', 'patent'): (patent2paper_target, patent2paper_source),
        ('patent', 'cites', 'patent'): (patent2patent_source, patent2patent_target),
        ('patent', 'cited by', 'patent'): (patent2patent_target, patent2patent_source)
    }

def get_cooccur_edata(data_path: str) -> torch.Tensor:
    print('get co-occur timestamps')
    result = []
    for year in range(2010, 2024):
        data_tmp = pd.read_excel(data_path + f'{year}/gephi_data.xlsx', sheet_name='edge')
        num_edge = len(list(data_tmp['Weight']))
        result += num_edge * [year]
    print('finish')
    return torch.tensor(result)

def get_patent_ipc_edata(data_path: str) -> torch.Tensor:
    print('get patent-ipc timestamps')
    result = []
    data = pd.read_excel(data_path + 'data.xlsx')
    data['IPC'] = data['IPC'].apply(lambda x: x.split('; '))
    for year in range(2010, 2024):
        data_tmp = data[data['申请日'] == year]
        edge_num_tmp = len(list(chain(*list(data_tmp['IPC']))))
        result += edge_num_tmp * [year]
    print('finish')
    return torch.tensor(result)

def get_patent_paper_edata(data_path: str) -> torch.Tensor:
    print('get patent-paper timestamps')
    patents = pd.read_excel(data_path + 'data.xlsx')
    papers = pd.read_excel(data_path + 'papers.xlsx')

    patent2year = dict(zip(list(patents['公开（公告）号']), list(patents['申请日'])))

    papers['year'] = papers['Matched Publication Numbers'].apply(lambda x: list(map(lambda p: patent2year[p], x.split(', '))))
    result = list(chain(*list(papers['year'])))
    print('finish')
    return torch.tensor(result)

def get_patent_patent_edata(data_path: str, patent2id: dict) -> torch.Tensor:
    print('get patent2patent data')
    data1 = pd.read_excel(data_path + '1-10000.xlsx', usecols=['公开（公告）号', '申请日', '引证专利', 'IPC']).rename(
        columns={'公开（公告）号': 'patent', '引证专利': 'inf_patent'})
    data2 = pd.read_excel(data_path + '10001-13517.xlsx', usecols=['公开（公告）号', '申请日', '引证专利', 'IPC']).rename(
        columns={'公开（公告）号': 'patent', '引证专利': 'inf_patent'})
    data = pd.concat([data1, data2]).dropna(subset=['IPC'])
    data['申请日'] = data['申请日'].apply(lambda x: int(x.year))
    result = list(data['申请日'])
    data = data.dropna(subset=['patent', 'inf_patent'], how='any')

    def inf_filter(inf_patents):
        inf_set = set(inf_patents.split('; '))
        patent_set = set(list(patent2id.keys()))

        if len(inf_set.intersection(patent_set)):
            return list(inf_set.intersection(patent_set))
        else:
            return None

    data['inf_patent'] = data['inf_patent'].apply(inf_filter)
    data = data.dropna(subset=['patent', 'inf_patent'], how='any')

    for index, row in data.iterrows():
        result += len(row['inf_patent']) * [row['申请日']]

    print('finish')
    return torch.tensor(result)

def prepare_edata(processed_data_path: str, ori_data_path: str, patent2id:dict) -> dict:
    print('2.prepare edata:')
    cooccur_edata = get_cooccur_edata(processed_data_path)
    patent2ipc_edata = get_patent_ipc_edata(ori_data_path)
    patent2paper_edata = get_patent_paper_edata(ori_data_path)
    patent2patent_edata = get_patent_patent_edata(ori_data_path, patent2id)
    print('2.edata prepared')
    return {
        ('ipc', 'co-occurs with', 'ipc'): cooccur_edata,
        ('ipc', 'assigned to', 'patent'): patent2ipc_edata,
        ('patent', 'assigned with', 'ipc'): patent2ipc_edata,
        ('patent', 'cites', 'paper'): patent2paper_edata,
        ('paper', 'cited by', 'patent'): patent2paper_edata,
        ('patent', 'cites', 'patent'): patent2patent_edata,
        ('patent', 'cited by', 'patent'): patent2patent_edata
    }

def prepare_ndata(processed_data_path: str) -> dict:
    print('3.prepare node feature matrices')

    ipc_embed_matrix = torch.load(processed_data_path + 'ipc_embed_matrix1.pt')
    patent_embed_matrix = torch.load(processed_data_path + 'patent_embed_matrix.pt')
    paper_embed_matrix = torch.load(processed_data_path + 'paper_embed_matrix.pt')
    print('3.node feature matrices prepared')
    return {
        'ipc': ipc_embed_matrix,
        'patent': patent_embed_matrix,
        'paper': paper_embed_matrix
    }

def prepare_temporal_ndata(graph: dgl.DGLGraph, processed_data_path: str, num_timestamps: int) -> dict:
    print('3.prepare node feature matrices')

    ipc_embed_matrix = torch.load(processed_data_path + 'ipc_embed_matrix1.pt')
    patent_embed_matrix = torch.randn([graph.num_nodes(ntype='patent'), num_timestamps, 768])
    paper_embed_matrix = torch.randn([graph.num_nodes(ntype='paper'), num_timestamps, 768])
    print('3.node feature matrices prepared')
    return {
        'ipc': ipc_embed_matrix,
        'patent': patent_embed_matrix,
        'paper': paper_embed_matrix
    }

if __name__ == '__main__':
    ori_data_path = '../data/大模型/'
    processed_data_path = '../result/大模型/'
    result_path = '../result/大模型/'

    with open(processed_data_path + 'ipc2id.json', 'r') as f:
        ipc2id = json.load(f)
        f.close()

    with open(processed_data_path + 'patent2id.json', 'r') as f:
        patent2id = json.load(f)
        f.close()

    with open(processed_data_path + 'paper2id.json', 'r') as f:
        paper2id = json.load(f)
        f.close()

    graph_data = prepare_graph_data(processed_data_path, ori_data_path, ipc2id, patent2id, paper2id)
    thg = dgl.heterograph(graph_data)

    edata = prepare_edata(processed_data_path, ori_data_path, patent2id)
    # ndata = prepare_ndata(processed_data_path)
    ndata = prepare_temporal_ndata(thg, processed_data_path, num_timestamps=14)
    '''metapaths = {
        'co-occur': [('ipc', 'assigned to', 'patent'), ('patent', 'assigned with', 'ipc')],
        'similar patent': [('ipc', 'assigned to', 'patent'), ('patent', 'cites', 'patent'), ('patent', 'assigned with', 'ipc')],
        'same paper': [('ipc', 'assigned to', 'patent'), ('patent', 'cites', 'paper'), ('paper', 'cited by', 'patent'), ('patent', 'assigned with', 'ipc')]
    }

    transform = dgl.AddMetaPaths(metapaths, keep_orig_edges=True)'''
    thg.edata['t'] = edata
    thg.ndata['f'] = ndata
    # thg = dgl.add_self_loop(thg, etype=('patent', 'cites', 'patent'))
    # thg = dgl.add_self_loop(thg, etype=('patent', 'cited by', 'patent'))
    # thg = dgl.remove_self_loop(thg, etype='co-occurs with')
    # thg = dgl.remove_self_loop(thg, etype=('patent', 'cites', 'patent'))
    '''new_thg = transform(thg)
    new_thg = dgl.remove_self_loop(new_thg, etype='co-occur')
    print(new_thg)'''
    dgl.save_graphs(result_path + 'thg1.dgl', [thg])

