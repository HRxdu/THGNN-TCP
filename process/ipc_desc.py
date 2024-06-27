import os
import json
import pandas as pd

def get_ipc_path(ipc_tree, ipc: str) -> list:
    result = [ipc]
    while ipc_tree[ipc]['value'] != 'root':
        result.append(ipc_tree[ipc]['parent'])
        ipc = ipc_tree[ipc]['parent']

    result.remove('root')
    result.reverse()
    return result

def gen_static_desc(id2ipc:dict, ipc_trees:dict, desc_datas:dict):
    ipc_data = list(id2ipc.values())
    print(len(ipc_data))
    result = {}
    for ipc in ipc_data:
        desc = ''
        for ipc_tree in ipc_trees.values():
            if ipc in ipc_tree.keys():
                ipc_path = get_ipc_path(ipc_tree, ipc)

        for desc_data in desc_datas.values():
            if ipc in desc_data.keys():
                for node in ipc_path:
                    desc += desc_data[node] + '; '

        if desc == '':
            ipc_path = [ipc[0], ipc[0:3], ipc[0:4]]
            for desc_data in desc_datas:
                if ipc_path[-1] in desc_data.keys():
                    for node in ipc_path:
                        desc += desc_data[node] + '; '

        result[ipc] = desc.strip('; ')

    with open(result_path + 'ipc_desc.json', 'w') as f:
        json.dump(result, f, indent=2)

def gen_temporal_desc(id2ipc:dict, ipc_trees:dict, desc_datas:dict):
    ipc_data = list(id2ipc.values())
    print(len(ipc_data))
    result = {}
    for ipc in ipc_data:
        result[ipc] = {}
        for year in range(2010, 2024):
            desc = ''
            result[ipc][year] = desc
            ipc_tree = ipc_trees[year]
            desc_data = desc_datas[year]
            if ipc in ipc_tree.keys():
                ipc_path = get_ipc_path(ipc_tree, ipc)
                if ipc in desc_data.keys():
                    for node in ipc_path:
                        desc += desc_data[node] + '; '

            if desc == '':
                ipc_path = [ipc[0], ipc[0:3], ipc[0:4]]
                if ipc_path[-1] in desc_data.keys():
                    for node in ipc_path:
                        desc += desc_data[node] + '; '

            result[ipc][year] = desc.strip('; ')

    with open(result_path + 'ipc_desc1.json', 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == '__main__':
    desc_data_path = '../data/'
    ipc_tree_data_path = '../result/大模型/'
    ipc_data_path = '../result/大模型/'
    result_path = '../result/大模型/'

    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    ipc_trees = {}
    desc_datas = {}

    for year in range(2024, 2009, -1):
        with open(ipc_tree_data_path + f'{year}/ipc_tree.json', 'r') as f:
            ipc_tree = json.load(f)
            f.close()
        ipc_trees[year] = ipc_tree

    for year in range(2024, 2009, -1):
        desc_data = pd.DataFrame(columns=['ipc', 'desc'])
        desc_files = os.listdir(desc_data_path + f'{year}')
        for file in desc_files:
            if file.endswith('txt'):
                data_tmp = pd.read_csv(desc_data_path + f'{year}/{file}', sep='\t', header=None).rename(
                    columns={0: 'ipc', 1: 'desc'})
                desc_data = pd.concat([desc_data, data_tmp])
        desc_data = dict(zip(desc_data['ipc'], desc_data['desc']))
        desc_datas[year] = desc_data

    with open(ipc_data_path + 'id2ipc.json', 'r') as f:
        id2ipc = json.load(f)
        f.close()

    # gen_static_desc(id2ipc, ipc_trees, desc_datas)
    gen_temporal_desc(id2ipc, ipc_trees, desc_datas)
    '''for year in range(2010, 2023):
            ipc_data = pd.read_excel(ipc_data_path + f'{year}/gephi_data.xlsx', sheet_name='node')['Id']
            ipc_data = list(map(ipc_transform, ipc_data))

            result = {}
            for ipc in ipc_data:
                desc = ''
                for ipc_tree in ipc_trees:
                    if ipc in ipc_tree.keys():
                        ipc_path = get_ipc_path(ipc_tree, ipc)

                for desc_data in desc_datas:
                    if ipc in desc_data.keys():
                        for node in ipc_path:
                            desc += desc_data[node] + '; '

                if desc == '':
                    ipc_path = [ipc[0], ipc[0:3], ipc[0:4]]
                    for desc_data in desc_datas:
                        if ipc_path[-1] in desc_data.keys():
                            for node in ipc_path:
                                desc += desc_data[node] + '; '

                result[ipc] = desc.strip('; ')

            with open(result_path + f'{year}/ipc_desc.json', 'w') as f:
                json.dump(result, f, indent=2)'''


