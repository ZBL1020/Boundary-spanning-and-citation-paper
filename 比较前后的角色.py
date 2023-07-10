import pandas as pd
from glob import glob

path = r'D:\郑碧丽\IPM论文数据\相似度计算结果\NLP\置信区间\*'
ori = open(r'D:\郑碧丽\IPM论文数据\相似度计算结果\NLP\NLP_role1.0.csv','r',encoding='utf-8')
o_f = pd.read_csv(ori,low_memory=False)
o_d = o_f.loc[:,["no","role"]]
files = glob(path)
for file in files:
    with open(file) as f:
        fin = pd.read_csv(f,low_memory=False)
        data = fin.loc[:,["no","role"]]
        for k, v in data.items():
            if v != o_d.loc[k,["no","role"]]:
                print()

