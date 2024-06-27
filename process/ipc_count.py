import pandas as pd
from itertools import chain
def ipc_count(data, result_path):
    ipc_num = []
    patent_num = []
    edge_num = []
    max_weight = []

    for year in range(2010, 2023):
        data_tmp = data[data['申请日'] == year]
        gephi_data_tmp = pd.read_excel(result_path + str(year) + '/gephi_data.xlsx', sheet_name='edge')

        ipc_tmp = list(data_tmp['IPC'].apply(lambda x: x.split('; ')))
        ipc_tmp = set(list(chain(*ipc_tmp)))
        ipc_num.append(len(ipc_tmp))

        patent_tmp = list(set(data_tmp['公开（公告）号']))
        patent_num.append(len(patent_tmp))

        edge_tmp = list(gephi_data_tmp['Weight'])
        edge_num.append(len(edge_tmp))
        max_weight.append(max(edge_tmp))

    result = pd.DataFrame({
        '申请日': list(range(2010, 2023)),
        '专利数': patent_num,
        'IPC数': ipc_num,
        '连边数': edge_num,
        '最大边权重': max_weight
    }, columns=['申请日', '专利数', 'IPC数', '连边数', '最大边权重'])
    result.to_excel(result_path + 'ipc_count.xlsx', index=None)


if __name__ == '__main__':
    data_path = "../data/大模型/"
    result_path = "../result/大模型/"

    data = pd.read_excel(data_path + 'data.xlsx')
    ipc_count(data, result_path)