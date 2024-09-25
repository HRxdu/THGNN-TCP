import json
import functools
import pandas as pd

from itertools import chain
from collections import Counter
from ipc_cooccur import ipc_sort
from ipc_adj_matrix import ipc_transform

if __name__ == '__main__':
    data_path = '../data/大模型/'
    result_path = '../result/大模型/'

    data = pd.read_excel(data_path + 'data.xlsx')
    # data = pd.read_excel(data_path + 'IPC_data_LLM.xlsx').dropna(subset=['IPC'])
    data['IPC'] = data['IPC'].apply(lambda x: x.split('; '))
    ipcs = list(set(chain(*list(data['IPC']))))
    ipcs = list(map(ipc_transform, ipcs))
    ipcs.sort(key=functools.cmp_to_key(ipc_sort))
    ipcs_set = list(set(ipcs))
    ipcs_set.sort(key=ipcs.index)

    ipc2id = dict(zip(ipcs_set, list(range(len(ipcs_set)))))
    id2ipc = dict(zip(list(range(len(ipcs_set))), ipcs_set))

    with open(result_path + 'ipc2id.json', 'w') as f:
        json.dump(ipc2id, f, indent=2)
        f.close()

    with open(result_path + 'id2ipc.json', 'w') as f:
        json.dump(id2ipc, f, indent=2)
        f.close()
