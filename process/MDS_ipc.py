import torch
from sklearn.manifold import MDS
def make_symmetric(matrix):
    """
    将矩阵变为对称矩阵

    参数:
    - matrix: 输入矩阵

    返回:
    - 对称矩阵
    """
    return (matrix + matrix.t())

def mds_embedding(distance_matrix, n_components):
    """
    使用MDS将距离邻接矩阵进行降维

    参数:
    - distance_matrix: 距离邻接矩阵
    - n_components: 期望的输出维度

    返回:
    - 特征矩阵
    """
    # Make the distance matrix symmetric
    distance_matrix = make_symmetric(distance_matrix)

    # Replace torch.inf values with a large finite value
    finite_value = 1e9
    distance_matrix[torch.isinf(distance_matrix)] = finite_value

    # Normalize the distance matrix to avoid large values
    normalized_distance_matrix = distance_matrix / torch.max(distance_matrix)

    # 计算期望的输出维度
    output_dimensions = n_components

    # 使用MDS进行降维
    mds = MDS(n_components=output_dimensions, dissimilarity='precomputed')
    feature_matrix = mds.fit_transform(normalized_distance_matrix.numpy())

    return feature_matrix

if __name__ == '__main__':
    result_path = '../result/大模型/'
    all_ipc_adj_matrix = torch.load(result_path + f'/ipc_adj_matrix.pt')
    all_ipc_result_matrix = mds_embedding(all_ipc_adj_matrix, n_components=128)
    torch.save(torch.tensor(all_ipc_result_matrix), result_path + f'/MDs_matrix.pt')
    print(all_ipc_result_matrix.shape)

    for year in range(2010, 2024):
        ipc_adj_matrix = torch.load(result_path + f'{year}/ipc_adj_matrix.pt')
        ipc_result_matrix = mds_embedding(ipc_adj_matrix, n_components=128)
        torch.save(torch.tensor(ipc_result_matrix), result_path + f'{year}/MDs_matrix.pt')
        print(ipc_result_matrix.shape)

    print("MDS特征矩阵")
