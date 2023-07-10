import numpy as np
import pandas as pd
from scipy import stats

fin = open(r'D:\郑碧丽\IPM论文数据\相似度计算结果\NLP\NLP_ROLE.csv','r',encoding='utf-8')
fout = open(r'D:\郑碧丽\IPM论文数据\相似度计算结果\NLP\95_NLP_role.csv','w',encoding='utf-8',newline='')
dict = {"no":[],"95%_tr":[],"95%_cr":[],"role":[]}
data = pd.read_csv(fin,low_memory=False)
tr = data.loc[:,"tr"].tolist()
cr = data.loc[:,"cr"].tolist()

ave_tr = np.mean(tr)
ave_cr = np.mean(cr)
std_tr = stats.sem(tr)
std_cr = stats.sem(cr)

upper_tr = ave_tr+1.96*std_tr
upper_cr = ave_cr +1.96*std_cr
lower_tr = ave_tr-1.96*std_tr
lower_cr = ave_cr-1.96*std_cr


for k,v in data.loc[:,"no"].items():
    no = v
    value_tr = data.loc[k,"tr"]
    value_cr = data.loc[k,"cr"]
    if (value_tr > upper_tr or value_tr < lower_tr) and (value_cr > upper_cr or value_cr < lower_cr):
        dict["no"].append(v)
        dict["95%_tr"].append(value_tr)
        dict["95%_cr"].append(value_cr)
        if value_tr >= ave_tr and value_cr >= ave_cr:
            dict["role"].append("Disseminator")
        if value_tr >= ave_tr and value_cr <= ave_cr:
            dict["role"].append("Broker")
        if value_tr <= ave_tr and value_cr <= ave_cr:
            dict["role"].append("Outlier")
        if value_tr <= ave_tr and value_cr >= ave_cr:
            dict["role"].append("Trigger")
        else:
            continue

df_1 = pd.DataFrame(pd.DataFrame.from_dict(dict, orient='index').values.T, columns=list(dict.keys()))
df_1.to_csv(fout)

