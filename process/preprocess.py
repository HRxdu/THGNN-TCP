import pandas as pd
from itertools import chain

def merge_data(data_path):
    d1 = pd.read_excel(data_path + '1-10000.xlsx', usecols=['公开（公告）号', 'IPC', '申请日'])
    d2 = pd.read_excel(data_path + '10001-13517.xlsx', usecols=['公开（公告）号', 'IPC', '申请日'])
    '''d1 = pd.read_excel(data_path, usecols=['公开（公告）号', 'IPC', '申请日'])
    d2 = pd.read_excel(data_path, usecols=['公开（公告）号', 'IPC', '申请日'])
    d3 = pd.read_excel(data_path, usecols=['公开（公告）号', 'IPC', '申请日'])
    d4 = pd.read_excel(data_path, usecols=['公开（公告）号', 'IPC', '申请日'])'''

    # data = pd.concat([d1, d2, d3, d4]).reset_index()
    data = pd.concat([d1, d2]).reset_index()
    data['申请日'] = data['申请日'].apply(lambda x: str(x.year))
    data = data[data['申请日'] >= '2010'].dropna(subset=['申请日', 'IPC'])
    data.to_excel(data_path + 'data.xlsx', index=None)

def ipc_count(data, result_path):
    ipc_num = []

    for year in range(2010, 2023):
        data_tmp = data[data['申请日'] == year]
        ipc_tmp = list(data_tmp['IPC'].apply(lambda x: x.split('; ')))
        ipc_tmp = set(list(chain(*ipc_tmp)))
        ipc_num.append(len(ipc_tmp))

    result = pd.DataFrame({
        '申请日': list(range(2010, 2023)),
        'IPC数': ipc_num
    }, columns=['申请日', 'IPC数'])
    result.to_excel(result_path + 'ipc_count.xlsx', index=None)


if __name__ == '__main__':
    data_path = "../data/大模型/"
    result_path = "../result/大模型/"

    merge_data(data_path)

    data = pd.read_excel(data_path + 'data.xlsx')
    ipc_count(data, result_path)


