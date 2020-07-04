import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

data=pd.read_excel('Breast_Cancer_Data.xlsx')

X = data.ix[:,0:9]

y = data.ix[:,9:]


model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(9).plot(kind='barh')

csfont = {'fontname':'Times New Roman'}
plt.yticks(fontname = "Times New Roman",fontsize=14)
plt.xticks(fontname = "Times New Roman",fontsize=14)
plt.xlabel("Feature importance",fontsize=14,**csfont)
plt.ylabel("Features",fontsize=14,**csfont)
#pie=plt.pie(model.feature_importances_,radius = 1, startangle=90,autopct='%1.1f%%')
#plt.legend(labels=X.columns, loc=9,bbox_to_anchor=(-0.1, 1.))
#plt.tight_layout()
#plt.title("Feature selection using Extra Tree Classifier",**csfont, fontsize=14)
plt.show()
