import os
import json

def dfs(parent, tree_tmp, node_tmp):
    result[node_tmp] = {
        'value': node_tmp,
        'parent': parent
    }

    if tree_tmp.keys():
        for child in tree_tmp.keys():
            dfs(parent=node_tmp, tree_tmp=tree_tmp[child], node_tmp=child)
    else:
        return

if __name__ == '__main__':
    classes = ['A', 'B', 'C', 'E', 'F', 'G', 'H']
    for i in range(2010, 2025):

        result_path = f'../result/大模型/{i}/'
        if not os.path.exists(result_path):
            os.mkdir(result_path)

        with open(f'../data/{i}/IPC hierarchy-{i}.json', 'r') as f:
            complete_tree = json.load(f)

        result = {
            'root': {
                'value': 'root',
                'parent': None
            }
        }

        for c in classes:
            dfs(parent='root', tree_tmp=complete_tree[c], node_tmp=c) #递归

            '''
            循环嵌套
            
            result[c] = {
                'value': c,
                'parent': 'root'
            }
            
            tree_c = complete_tree[c]
    
            for node1 in tree_c.keys():
                result[node1] = {
                    'value': node1,
                    'parent': c
                }
    
                tree_n1 = tree_c[node1]
    
                for node2 in tree_n1.keys():
                    result[node2] = {
                        'value': node2,
                        'parent': node1
                    }
    
                    tree_n2 = tree_n1[node2]
    
                    for node3 in tree_n2.keys():
                        result[node3] = {
                            'value': node3,
                            'parent': node2
                        }
    
                        tree_n3 = tree_n2[node3]
    
                        for node4 in tree_n3.keys():
                            result[node4] = {
                                'value': node4,
                                'parent': node3
                            }
    
                            tree_n4 = tree_n3[node4]
    
                            for node5 in tree_n4.keys():
                                result[node5] = {
                                    'value': node5,
                                    'parent': node4
                                }
    
                                tree_n5 = tree_n4[node5]
    
                                for node6 in tree_n5.keys():
                                    result[node6] = {
                                        'value': node6,
                                        'parent': node5
                                    }
    
                                    tree_n6 = tree_n5[node6]
    
                                    for node7 in tree_n6.keys():
                                        result[node7] = {
                                            'value': node7,
                                            'parent': node6
                                        }
    
                                        tree_n7 = tree_n6[node7]
    
                                        for node8 in tree_n7.keys():
                                            result[node8] = {
                                                'value': node8,
                                                'parent': node7
                                            }
    
                                            tree_n8 = tree_n7[node8]
    
                                            for node9 in tree_n8.keys():
                                                result[node9] = {
                                                    'value': node9,
                                                    'parent': node8
                                                }
    
                                                tree_n9 = tree_n8[node9]
    
                                                for node10 in tree_n9.keys():
                                                    result[node10] = {
                                                        'value': node10,
                                                        'parent': node9
                                                    }
    
                                                    tree_n10 = tree_n9[node10]
    
                                                    for node11 in tree_n10.keys():
                                                        result[node11] = {
                                                            'value': node11,
                                                            'parent': node10
                                                        }
    
                                                        tree_n11 = tree_n10[node11]
    
                                                        for node12 in tree_n11.keys():
                                                            result[node12] = {
                                                                'value': node12,
                                                                'parent': node11
                                                            }
    
                                                            tree_n12 = tree_n11[node12]'''

        with open(result_path + 'ipc_tree.json', 'w') as f:
            json.dump(result, f, indent=2)
