import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
#data = pd.read_csv("LandSlide_Data.csv")
#X = data.iloc[:,0:5]  #independent columns
#y = data.iloc[:,5:0]    #target column i.e price range

data=pd.read_excel('Breast_Cancer_Data.xlsx')

X = data.ix[:,0:9]

y = data.ix[:,9:]

#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=7)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(9,'Score'))  #print 8 best features
feature=pd.Series(fit.scores_ , index=X.columns)
feature.nlargest(9).plot(kind='barh')
#plt.tight_layout()

#pie=plt.pie(fit.scores_,radius = 1, startangle=90,autopct='%1.1f%%')
#plt.legend(labels=X.columns, loc=9,bbox_to_anchor=(-0.1, 1.))
csfont = {'fontname':'Times New Roman'}
plt.yticks(fontname = "Times New Roman",fontsize=14,fontweight='bold')
plt.xticks(fontname = "Times New Roman",fontsize=14,fontweight='bold')
plt.xlabel("Feature importance",fontsize=14,fontweight='bold',**csfont)
plt.ylabel("Features",fontsize=14,fontweight='bold',**csfont)
#plt.title("Feature selection using Chi Square",**csfont, fontsize=14)
plt.show()
