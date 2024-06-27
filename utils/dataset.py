import dgl
from torch.utils.data import Dataset
from dgl.dataloading.negative_sampler import GlobalUniform

class LLMDataset_edge(Dataset):
    def __init__(self, graph_path, args):
        graph = dgl.load_graphs(graph_path)[0][0]
        for etype in graph.canonical_etypes:
            filtered_edges = graph.filter_edges(lambda x: self.edge_filter(x, t=args.test_timestamp), etype=etype)
            graph.remove_edges(eids=filtered_edges, etype=etype)

        self.target_graph = dgl.edge_type_subgraph(graph, etypes=['co-occurs with'])
        self.neg_sampler = GlobalUniform(k=1)

    def __getitem__(self, idx):
        pos_pairs = self.target_graph.edges('uv')
        neg_pairs = self.neg_sampler(self.target_graph, eids=self.target_graph.edge_ids(pos_pairs[0], pos_pairs[1]))

        return pos_pairs[0][idx], pos_pairs[1][idx], neg_pairs[0][idx], neg_pairs[1][idx]

    def __len__(self):
        return self.target_graph.num_edges()

    def edge_filter(self, edges, t):
        return edges.data['t'] == t

class LLMDataset_node(Dataset):
    def __init__(self, graph_path, args):
        graph = dgl.load_graphs(graph_path)[0][0]
        for etype in graph.canonical_etypes:
            filtered_edges = graph.filter_edges(lambda x: self.edge_filter(x, t=args.test_timestamp), etype=etype)
            graph.remove_edges(eids=filtered_edges, etype=etype)

        self.target_graph = dgl.edge_type_subgraph(graph, etypes=['co-occurs with'])

    def __getitem__(self, idx):
        return self.target_graph.nodes(ntype='ipc')[idx]

    def __len__(self):
        return self.target_graph.num_nodes(ntype='ipc')

    def edge_filter(self, edges, t):
        return edges.data['t'] == t

class LLMDataset_eval(Dataset):
    def __init__(self, pos_pairs, neg_pairs):
        self.pos_u = pos_pairs[:, 0]
        self.pos_v = pos_pairs[:, 1]
        self.neg_u = neg_pairs[:, 0]
        self.neg_v = neg_pairs[:, 1]

    def __getitem__(self, idx):
        return self.pos_u[idx], self.pos_v[idx], self.neg_u[idx], self.neg_v[idx]

    def __len__(self):
        return self.pos_u.shape[0]

class LLMDataset_t(Dataset):
    def __init__(self, graph_path, timestamp):
        self.timestamp = timestamp
        self.graph_path = graph_path
        self.graph = dgl.load_graphs(graph_path)[0][0]
        self.nodes = self.graph.nodes(ntype='ipc')
        for etype in self.graph.canonical_etypes:
            filtered_edges = self.graph.filter_edges(lambda x: self.edge_filter4not_now(x, t=self.timestamp), etype=etype)
            self.graph.remove_edges(eids=filtered_edges, etype=etype)
        '''co_occur_metapath = {
            'co-occur': [('ipc', 'assigned to', 'patent'), ('patent', 'assigned with', 'ipc')]
        }
        transform = dgl.AddMetaPaths(co_occur_metapath, keep_orig_edges=True)
        self.new_graph_t = transform(self.graph)'''
        self.target_graph_t = dgl.edge_type_subgraph(self.graph, etypes=['co-occurs with'])
        # self.target_graph_t = dgl.edge_type_subgraph(self.graph, etypes=['co-occur'])
        # self.target_graph_t.remove_edges(eids=self.target_graph_t.filter_edges(lambda x: self.edge_filter4not_now(x, t=self.timestamp)))
        self.neg_sampler = GlobalUniform(k=1)

    def __getitem__(self, idx):
        pos_pairs = self.target_graph_t.edges('uv')
        neg_pairs = self.neg_sampler(self.target_graph_t, eids=self.target_graph_t.edge_ids(pos_pairs[0], pos_pairs[1]))

        return pos_pairs[0][idx], pos_pairs[1][idx], neg_pairs[0][idx], neg_pairs[1][idx]

    def __len__(self):
        return self.target_graph_t.num_edges()

    def edge_filter4after(self, edges, t):
        return edges.data['t'] > t

    def edge_filter4not_now(self, edges, t):
        return edges.data['t'] != t

    def get_graph_t(self):
        graph = self.graph
        for etype in graph.canonical_etypes:
            filtered_edges = graph.filter_edges(lambda x: self.edge_filter4not_now(x, t=self.timestamp), etype=etype)
            graph.remove_edges(eids=filtered_edges, etype=etype)
        return graph

# dataset = LLMDataset(graph_path='../result/大模型/thg.dgl', timestamp=2023)
