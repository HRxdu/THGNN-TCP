import json
import pickle
import networkx as nx
import pandas as pd

from collections import Counter

def ipc_retransform(ipc: str):
    # G06F0003048860 -> G06F3/04886
    part1 = ipc[:4]
    part2 = ipc[4:8].lstrip('0')
    part3 = ipc[8:].rstrip('0')
    if part3 == '':
        part3 = '00'
    result = part1 + part2 + '/' + part3
    return result

def add_weighted_edge(G: nx.Graph, u, v, weight):
    # 检查边是否已存在
    if G.has_edge(u, v):
        # 如果已存在，则获取当前权重并加上新权重
        current_weight = G[u][v]['weight']
        new_weight = current_weight + weight
        G[u][v]['weight'] = new_weight
    else:
        # 如果不存在，则直接添加边及权重
        G.add_edge(u, v, weight=weight)

def main(data, num_edge, result_path, ipc_level: str, topn=10):
    print(ipc_level + 50 * '-')
    if ipc_level == 'section':
        data['src'] = data['src'].apply(lambda x: x[0])
        data['dst'] = data['dst'].apply(lambda x: x[0])
    if ipc_level == 'class':
        data['src'] = data['src'].apply(lambda x: x[:3])
        data['dst'] = data['dst'].apply(lambda x: x[:3])
    if ipc_level == 'subclass':
        data['src'] = data['src'].apply(lambda x: x[:4])
        data['dst'] = data['dst'].apply(lambda x: x[:4])
    if ipc_level == 'field':
        data['src'] = data['src'].apply(lambda x: x[:x.find('/')])
        data['dst'] = data['dst'].apply(lambda x: x[:x.find('/')])

    ipcs = list(data['src']) + list(data['dst'])
    ipcs = list(set(ipcs))
    ipcs.sort()
    print(ipcs)

    num_ipc = len(ipcs)
    print(f'num_{ipc_level}:{num_ipc}')

    edges = list(zip(list(data['src']), list(data['dst'])))

    G = nx.Graph()
    for edge in edges:
        add_weighted_edge(G, u=edge[0], v=edge[1], weight=1)
    edges = G.edges(data=True)
    edges = list(map(lambda x: tuple([x[0], x[1], x[2]['weight']]), edges))
    print(edges)

    edges = list(map(lambda x: tuple([(x[0] + '-' + x[1]), x[2]]), edges))
    edges.sort(key=lambda x: x[1], reverse=True)

    # edges = pd.DataFrame(edges[:topn], columns=['edge', 'weight'])
    edges = pd.DataFrame(edges, columns=['edge', 'weight'])
    edges['percentage'] = edges['weight'] / num_edge
    print(edges)
    edges.to_excel(result_path + f'{ipc_level}.xlsx', index=None)

if __name__ == '__main__':
    levels = ['section', 'class', 'subclass', 'field', 'last']
    result_path = '预测分析/'
    p = 0.85

    with open('result/大模型/id2ipc.json', 'r') as f:
        id2ipc = json.load(f)
        f.close()

    for level in levels:

        with open(f'result/大模型/prediction_analysis_p={p}.pkl', 'rb') as f:
            data = pickle.load(f)
            data = pd.DataFrame(data, columns=['src', 'dst'])
            print(len(data))
            f.close()

        data['src'] = data['src'].apply(lambda x: id2ipc[str(x)])
        data['dst'] = data['dst'].apply(lambda x: id2ipc[str(x)])

        data['src'] = data['src'].apply(ipc_retransform)
        data['dst'] = data['dst'].apply(ipc_retransform)

        main(data, num_edge=len(data), result_path=result_path, ipc_level=level, topn=5)
        print('\n')
