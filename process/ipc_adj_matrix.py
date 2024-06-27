import json
import torch
import pandas as pd
from itertools import chain

def ipc_transform(ipc: str) -> str:
    # A61B5/145 -> A61B0005145000
    result = ipc[:4]
    symbol_idx = ipc.find('/')
    part1 = ipc[4:symbol_idx]
    part2 = ipc[symbol_idx+1:]

    result += (4 - len(part1)) * '0' + part1 + part2 + (6 - len(part2)) * '0'
    return result

'''def id_ipc_mapping(data_path: str):
    ipc = []
    for i in range(2010, 2023):
        data_tmp = pd.read_excel(data_path + str(i) + '/gephi_data.xlsx', sheet_name='node')
        ipc += list(data_tmp['Id'])
    ipc = list(set(ipc))
    ipc.sort(key=functools.cmp_to_key(ipc_sort))
    ipc = list(map(ipc_transform, ipc))

    id2ipc = dict(zip(list(range(len(ipc))), ipc))
    with open('../result/大模型/id2ipc.json', 'w') as f:
        json.dump(id2ipc, f, indent=2)

    ipc2id= dict(zip(ipc, list(range(len(ipc)))))
    with open('../result/大模型/ipc2id.json', 'w') as f:
        json.dump(ipc2id, f, indent=2)'''

def ipc_semantic_distance(ipc_trees: list, ipc1: str, ipc2: str):
    err_flag = 0

    def get_ipc_path(ipc_tree, ipc: str) -> list:
        result = [ipc]
        while ipc_tree[ipc]['value'] != 'root':
            result.append(ipc_tree[ipc]['parent'])
            ipc = ipc_tree[ipc]['parent']

        result.reverse()
        return result

    def find_last_common_parent(ipc1path: list, ipc2path: list) -> str:
        # AKA lcp
        len1 = len(ipc1path)
        len2 = len(ipc2path)

        result = ''
        for i in range(min(len1, len2)):
            if ipc1path[i] == ipc2path[i]:
                result = ipc1path[i]

        return result

    def get_lcp_path(lcp: str) -> list:
        result = [lcp]
        while ipc_tree[lcp]['value'] != 'root':
            result.append(ipc_tree[lcp]['parent'])
            lcp = ipc_tree[lcp]['parent']

        result.reverse()
        return result

    ipc_tree = {}
    for i in ipc_trees:
        if ipc1 in i.keys() and ipc2 in i.keys():
            ipc_tree = i
            break

    if ipc_tree == {}:
        err_flag = 1
        return torch.inf, err_flag

    ipc1path = get_ipc_path(ipc_tree, ipc1)
    ipc2path = get_ipc_path(ipc_tree, ipc2)
    lcp = find_last_common_parent(ipc1path, ipc2path)
    lcp_path = get_lcp_path(lcp)
    return len(ipc1path) + len(ipc2path) - 2 * len(lcp_path), err_flag

def gen_ipc_adj_matrix(data, ipc_trees, log_file):
    print(f'year:{year}', file=log_file)
    ipc_tmp = list(set(chain(*list(data['IPC']))))
    ipc_num = len(ipc_tmp)

    # ipc_num = len(list(id2ipc_dict.keys()))
    result = torch.zeros([ipc_num, ipc_num])
    err_pairs = []
    for i in range(ipc_num):
        for j in range(i+1, ipc_num):
            '''try:
                result[i][j] = ipc_semantic_distance(ipc_trees, ipc_tmp[i], ipc_tmp[j])
            except KeyError as e:
                err_ipc.append(str(e).strip('\''))
                continue'''
            result[i][j], err_flag = ipc_semantic_distance(ipc_trees, ipc_tmp[i], ipc_tmp[j])
            if err_flag:
                err_pairs.append((ipc_tmp[i], ipc_tmp[j]))


    print(f'number of ipc:{ipc_num}', file=log_file)
    print(f'number of ipc pairs:{ipc_num ** 2}', file=log_file)
    print(f'number of error ipc pairs:{len(err_pairs)}', file=log_file)
    print('percentage of error ipc pairs:{:.2f}%'.format(len(err_pairs) / ipc_num ** 2 * 100), file=log_file)
    print('\n', file=log_file)
    return result

if __name__ == '__main__':
    ori_data_path = '../data/大模型/'
    processed_data_path = '../result/大模型/'
    result_path = '../result/大模型/'
    ipc_trees = []
    data = pd.read_excel(ori_data_path + 'data.xlsx')
    # data = pd.read_excel(ori_data_path + 'IPC_data_LLM.xlsx').dropna(subset=['IPC'])
    data['IPC'] = data['IPC'].apply(lambda x: list(map(ipc_transform, x.split('; '))))

    for year in range(2024, 2009, -1):
        with open(processed_data_path + f'{year}/ipc_tree.json', 'r') as f:
            ipc_tree = json.load(f)
            f.close()

        ipc_trees.append(ipc_tree)

    with open(result_path + 'ipc_adj_matrix.log', 'w') as f:
        for year in range(2010, 2024):
            data_tmp = data[data['申请日'] == year]
            ipc_adj_matrix = gen_ipc_adj_matrix(data_tmp, ipc_trees, log_file=f)
            torch.save(ipc_adj_matrix, result_path + f'{year}/ipc_adj_matrix.pt')
        f.close()