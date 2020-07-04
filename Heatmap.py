import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
data=pd.read_excel('Breast_Cancer_Data.xlsx')

X = data.ix[:,0:9]

y = data.ix[:,9:]


#get correlations of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(7,7))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn", linewidths=1)

csfont = {'fontname':'Times New Roman'}
plt.yticks(fontsize=12, **csfont)
plt.xticks(fontsize=12, **csfont)
#plt.title("(c) Feature selection using Heat Map")
plt.tight_layout()
plt.show()
