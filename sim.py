from sentence_transformers import util
import torch

embeddings1 = torch.load(r'E:\D盘\郑碧丽\IPM论文\相似度计算\语料\NLP\语料\目标文献1.pt',map_location='cpu')      #目标文献语料路径
for ir in range(1,122347):
    print(ir)
    path1=f'E:\\D盘\\郑碧丽\\IPM论文\\相似度计算\\语料\\NLP\\语料\\reference20220518\\{ir}.pt'
    path2=f'E:\\D盘\\郑碧丽\\IPM论文\\相似度计算\\语料\\NLP\\语料\\citation20220518\\{ir}.pt'
    # path1 = r'E:\D盘\郑碧丽\IPM论文\相似度计算\语料\NLP\语料\reference20220518' + str(ir) + '.pt'     #参考文献语料路径
    # path2 = r'E:\D盘\郑碧丽\IPM论文\相似度计算\语料\LIS\lis_citation\lis_c_' + str(ir) + '.pt'    #施引文献语料路径
    citation_emb = torch.load(path2,map_location='cpu')
    ref_emb = torch.load(path1,map_location='cpu')
    emb = embeddings1[ir-1]
    #Compute cosine-similarities for each sentence with each other sentence
    cosine_scores = util.pytorch_cos_sim(emb, citation_emb)
    cosine_scores1 = util.pytorch_cos_sim(citation_emb, ref_emb)
    #Find the pairs with the highest cosine similarity scores
    w_p = r'E:\D盘\郑碧丽\IPM论文\相似度计算\相似度计算txt\NLP\target-citation-20220519\NLP_'+str(ir)+'tc.txt'   #目标文献-施引文献
    w_p1 = r'E:\D盘\郑碧丽\IPM论文\相似度计算\相似度计算txt\NLP\reference-citation-20220519\NLP_'+str(ir)+'cf.txt'                 #参考文献-施引文献
    f = open(w_p,'w')
    f1 = open(w_p1,'w')
    for i in range(len(cosine_scores)):
        for j in range(len(cosine_scores[0])):
            f.write("{}\t\t{}\t\t{:.4f}\n".format(i, j, cosine_scores[i][j]))
    for i in range(len(cosine_scores1)):
        for j in range(len(cosine_scores1[0])):
            f1.write("{}\t\t{}\t\t{:.4f}\n".format(i, j, cosine_scores1[i][j]))
            #pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})