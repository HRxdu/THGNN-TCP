import torch

if __name__ == '__main__':
    result_path = '../result/大模型/'
    embed_matrix = torch.load(result_path + 'ipc_embed_matrix_be.pt')
    all_MDs_result_matrix = torch.load(result_path + f'/MDs_matrix.pt')
    # 沿着第二维拼接这两个张量
    concatenated_matrix = torch.cat((embed_matrix, all_MDs_result_matrix), dim=1)
    torch.save(concatenated_matrix, result_path + f'/ipc_embed_matrix.pt')

    embed_matrix_1 = torch.load(result_path + 'ipc_embed_matrix1_be.pt')
    all_MDs_result_matrix_expanded = all_MDs_result_matrix.unsqueeze(1)
    all_MDs_result_matrix_expanded = all_MDs_result_matrix_expanded.expand(-1, 14, -1)
    concatenated_matrix1 = torch.cat((embed_matrix_1, all_MDs_result_matrix_expanded), dim=2)
    torch.save(concatenated_matrix1, result_path + f'/ipc_embed_matrix1.pt')
