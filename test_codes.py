import torch
import math


def positional_encoding(num_indices, max_seq_len, d_model):
    """
    生成位置编码矩阵
    :param max_seq_len: 序列最大长度
    :param d_model: 模型的维度
    :return: 位置编码矩阵，形状为(max_seq_len, d_model)
    """
    pe = torch.zeros(num_indices, max_seq_len, d_model)  # 初始化位置编码矩阵
    position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)  # 位置索引 (0, 1, ..., max_seq_len-1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # 计算衰减项

    pe[:, :, 0::2] = torch.sin(position * div_term)  # 偶数维度使用sin函数
    pe[:, :, 1::2] = torch.cos(position * div_term)  # 奇数维度使用cos函数

    return pe


# 示例：生成一个序列长度为100，模型维度为512的位置编码
num_indices = 50
max_seq_length = 100
model_dim = 512
positional_encoding_matrix = positional_encoding(num_indices, max_seq_length, model_dim)
print("Positional Encoding Matrix Shape:", positional_encoding_matrix.shape)