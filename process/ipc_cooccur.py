import os
import functools
import pandas as pd
from itertools import combinations, chain
from collections import Counter

def ipc_sort(x:str, y:str):
    # 第一位
    if x[0] > y[0]:
        return 1
    elif x[0] < y[0]:
        return -1
    else:
        # 第四位
        if x[3] > y[3]:
            return 1
        elif x[3] < y[3]:
            return -1
        else:
            # 第2-3位
            if int(x[1:3]) > int(y[1:3]):
                return 1
            elif int(x[1:3]) < int(y[1:3]):
                return -1
            else:
                x_id = x.find('/')
                y_id = y.find('/')
                if x_id == -1:
                    return -1
                if y_id == -1:
                    return 1
                # 第4位到"/"之间
                if int(x[4:x_id]) > int(y[4:y_id]):
                    return 1
                elif int(x[4:x_id]) < int(y[4:y_id]):
                    return -1
                else:
                    # "/" 之后
                    if x[x_id+1:] > y[y_id+1:]:
                        return 1
                    elif x[x_id+1:] < y[y_id+1:]:
                        return -1
    return 0
def co_occur(data, result_path):
    for year in range(2010, 2024):
        print(year)
        if not os.path.exists(result_path + str(year)):
            os.mkdir(result_path + str(year))

        data_tmp = data[data['申请日'] == year]
        ipc = list(data_tmp['IPC'].apply(lambda x: x.split('; '))) # list of list
        edge = []

        node = list(set(chain(*ipc)))
        node.sort(key=functools.cmp_to_key(ipc_sort))

        df_node = pd.DataFrame({
            'Id': node,
            'Label': node
        })

        for i in ipc:
            if len(i) >= 2:
                edge += list(combinations(i, 2))

        co_occur_count = dict(Counter(edge))

        edge = list(map(lambda x: (x[0][0], x[0][1], x[1]), co_occur_count.items()))
        df_edge = pd.DataFrame(edge, columns=['Source', 'Target', 'Weight']).sort_values(by=['Weight'], ascending=False)

        with pd.ExcelWriter(result_path + str(year) + "/gephi_data.xlsx") as w:
            df_node.to_excel(w, sheet_name='node', index=None)
            df_edge.to_excel(w, sheet_name='edge', index=None)

if __name__ == '__main__':
    data_path = "../data/大模型/"
    result_path = "../result/大模型/"

    # data = pd.read_excel(data_path + 'data.xlsx')
    data = pd.read_excel(data_path + 'IPC_data_LLM.xlsx').dropna(subset=['IPC'])
    co_occur(data, result_path)