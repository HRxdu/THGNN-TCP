import json
import torch
import numpy as np

if __name__ == '__main__':
    data_path = '../result/大模型/'
    result_path = '../result/大模型/'

    patent_embed = np.load(data_path + 'patent_embeddings-2.npz', allow_pickle=True)
    patents = list(patent_embed.keys())
    print(len(patents))

    patent2id = dict(zip(patents, list(range(len(patents)))))
    id2patent = dict(zip(list(range(len(patents))), patents))

    with open(result_path + 'patent2id.json', 'w') as f:
        json.dump(patent2id, f, indent=2)
        f.close()

    with open(result_path + 'id2patent.json', 'w') as f:
        json.dump(id2patent, f, indent=2)
        f.close()

    patent_embed_matrix = torch.tensor(list(patent_embed.values()))
    print(patent_embed_matrix.shape)
    torch.save(patent_embed_matrix, result_path + 'patent_embed_matrix.pt')
