import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from econml.dml import LinearDML
from econml.cate_interpreter import SingleTreeCateInterpreter, SingleTreePolicyInterpreter
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor
from econml.dml import CausalForestDML

# 加载数据集
data_as = pd.read_csv(r'F:\D盘\郑碧丽\IPM论文\PSM\JOI补充数据\as.csv',encoding='utf-8')
# data_lis = pd.read_csv(path+'/lis.csv')
# data_nlp = pd.read_csv(path+'/nlp.csv')

# 整理数据
Y = data_as["n_citation"]  # 因变量
T = data_as['M'].apply(lambda x: 1 if x>=np.mean(data_as['M']) else 0)
W = data_as.drop(columns=["n_citation", "M", "ID"])  # 协变量
X = np.array(data_as[["M"]]).reshape(-1, 1) # 自变量，因为自变量是一维的，所以做了reshape

# 分数据集
Y_train, Y_val, T_train, T_val, X_train, X_val, W_train, W_val = train_test_split(Y, T, X, W, test_size=.2)

#可以实验以下模型：LinearDML；SparseLinearDML；DML；CausalForestDML
#X和Y的相关函数：model_y, X和T的相关函数：model_t
'''
est = LinearDML( model_y=GradientBoostingRegressor(),
    model_t=GradientBoostingRegressor(),
    featurizer=PolynomialFeatures(degree=2, include_bias=False),)
est.fit(Y_train, T_train, X=X_train,W=W_train, cache_values = True)
est.summary()
'''

forest_model = CausalForestDML(max_depth=3)
forest_model = forest_model.fit(Y=Y,T=T,X=X,W=W)
intrp = SingleTreeCateInterpreter(max_depth=2).interpret(forest_model,X)
intrp.plot(feature_names="M", fontsize=12)
# print(intrp)
