import numpy as np
import pandas as pd
from scipy.stats import f_oneway

fin = open(r'D:\郑碧丽\IPM论文数据\相似度计算结果\as\as_role1.0.csv','r',encoding='utf-8')
df = pd.read_csv(fin,low_memory=False)
role = df.groupby("role")["cr"].apply(list)
as_broker = role.loc["Broker"]
as_disseminator = role.loc["Disseminator"]
as_outlier = role.loc["Outlier"]
as_trigger = role.loc["Trigger"]

f,p =f_oneway(np.array(as_broker),np.array(as_outlier),np.array(as_trigger),np.array(as_disseminator))
print(f,p)





