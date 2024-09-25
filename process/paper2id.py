import json
import torch
import numpy as np
import pandas as pd

if __name__ == '__main__':
    ori_data_path = '../data/大模型/'
    processed_data_path = '../result/大模型/'
    result_path = '../result/大模型/'

    chn_papers = list(pd.read_excel(ori_data_path + '整理后的格式化中文.xlsx')['Title'])
    eng_papers = list(pd.read_excel(ori_data_path + '整理后的格式化英文.xlsx')['Title'])
    papers = chn_papers + eng_papers

    paper2id = dict(zip(papers, list(range(len(papers)))))
    id2paper = dict(zip(list(range(len(papers))), papers))

    with open(result_path + 'paper2id.json', 'w') as f:
        json.dump(paper2id, f, indent=2)
        f.close()

    with open(result_path + 'id2paper.json', 'w') as f:
        json.dump(id2paper, f, indent=2)
        f.close()

    chn_paper_vec = np.load(processed_data_path + '中文embedding.npz',)['arr_0']
    eng_paper_vec = np.load(processed_data_path + '英文embedding.npz')['arr_0']
    paper_embed_matrix = torch.tensor(np.concatenate([chn_paper_vec, eng_paper_vec]))
    torch.save(paper_embed_matrix, result_path + 'paper_embed_matrix.pt')
    print(paper_embed_matrix.shape)

