from transformers import AutoTokenizer, AutoModel
import pickle
import json
import torch
import torch.nn.functional as F

def gen_embed_matrix(ipc_desc, ipc_num, tokenizer, model):

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    result = torch.zeros([ipc_num, 768])

    for index, desc in enumerate(ipc_desc.values()):
        print(index)
        # Sentences we want sentence embeddings for
        sentences = desc.split('; ')

        # Tokenize sentences
        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        encoded_input = encoded_input.to(torch.device('cuda:0'))

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        sentence_embeddings = torch.mean(sentence_embeddings, dim=0)
        # print("Sentence embeddings:")
        # print(sentence_embeddings)
        result[index] = sentence_embeddings

    return result

def gen_temporal_embed_matrix(ipc_desc, ipc_num, tokenizer, model, time_range):

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    result = torch.zeros([ipc_num, len(time_range), 768])

    for index, time_corpus in enumerate(ipc_desc.values()):
        print(index)
        for tid, t in enumerate(time_range):
            sentences = time_corpus[str(t)]
            # Sentences we want sentence embeddings for

            # Tokenize sentences
            encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
            encoded_input = encoded_input.to(torch.device('cuda:0'))

            # Compute token embeddings
            with torch.no_grad():
                model_output = model(**encoded_input)

            # Perform pooling
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

            # Normalize embeddings
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            sentence_embeddings = torch.mean(sentence_embeddings, dim=0)
            # print("Sentence embeddings:")
            # print(sentence_embeddings)
            result[index][tid] = sentence_embeddings

    return result

if __name__ == '__main__':
    # Load model from local
    tokenizer = AutoTokenizer.from_pretrained('../model/all-mpnet-base-v2')
    model = AutoModel.from_pretrained('../model/all-mpnet-base-v2')
    model = model.to(torch.device('cuda:0'))

    data_path = '../result/大模型/'
    result_path = '../result/大模型/'

    '''with open(data_path + 'ipc_desc.json', 'r') as f:
        ipc_desc = json.load(f)
        f.close()'''

    with open(data_path + 'ipc_desc1.json', 'r') as f:
        ipc_desc = json.load(f)
        f.close()

    ipc_num = len(list(ipc_desc.keys()))
    print(ipc_num)
    # embed_matrix = gen_embed_matrix(ipc_desc, ipc_num, tokenizer, model)
    embed_matrix = gen_temporal_embed_matrix(ipc_desc, ipc_num, tokenizer, model, time_range=list(range(2010, 2024)))
    print(embed_matrix.shape)
    torch.save(embed_matrix, result_path + 'ipc_embed_matrix1.pt')
    '''with open(result_path + 'ipc_embed_matrix.pkl', 'wb') as f:
        pickle.dump(embed_matrix, f)
        f.close()'''
    '''for year in range(2010, 2023):
        print(year)
        with open(data_path + f'{year}/ipc_desc.json', 'r') as f:
            ipc_desc = json.load(f)
        ipc_num = len(list(ipc_desc.keys()))
        embed_matrix = gen_embed_matrix(ipc_desc, ipc_num)
        print(embed_matrix.shape)
        torch.save(embed_matrix, result_path + f'{year}/ipc_embed_matrix.pt')'''

