from sentence_transformers import SentenceTransformer
import torch
import time

model = SentenceTransformer(r'F:\郑碧丽\trained_model_1\1104999')
# Single list of sentences
c = 1
sen = open(r'E:\D盘\郑碧丽\IPM论文\相似度计算\摘要数据\LIS\mubiao\1-8706_r_ab.txt', 'r', encoding='utf-8').read().split('<---------->\n')
print(len(sen))

sentences = sen

embeddings_target_document  = model.encode(sentences, batch_size=6, convert_to_tensor=True)
filepath = 'E:\\D盘\\郑碧丽\\IPM论文\\相似度计算\\语料\\LIS\\target\\LIS' + str(c) + '.pt'
# filepath = 'D:\\郑碧丽\\AI数据集\\G0目标文献语料\\' + str(c) + '.pt'
torch.save(embeddings_target_document, filepath)

