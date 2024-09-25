import dgl

if __name__ == '__main__':
    dataset = '大模型'
    thg = dgl.load_graphs(f'result/{dataset}/thg1.dgl')[0][0]
    print(thg)
    graphs = dgl.load_graphs(f'result/{dataset}/graphs.dgl')[0]
    for idx, graph in enumerate(graphs):
        print(idx)
        print(graph)
        ipc, patent = map(lambda x: x.numpy().tolist(), graph.edges(etype='assigned to'))
        print(f'num ipc = {len(set(ipc))}')
        print(f'num patent = {len(set(patent))}')
        paper, _ = map(lambda x: x.numpy().tolist(), graph.edges(etype=('paper', 'cited by', 'patent')))
        print(f'num paper = {len(set(paper))}')