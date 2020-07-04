import warnings
warnings.filterwarnings("ignore")
from sklearn import metrics
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import pylab
import time
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

start =time.time()
#reading the data
data1=pd.read_excel('Breast_Cancer_Data.xlsx')
data2=pd.read_excel('Breast_Cancer_Data_Eliminating_Mitoses.xlsx')
data3=pd.read_excel('Breast_Cancer_Data_Eliminating_SingleEpithelialCell.xlsx')
data4=pd.read_excel('Breast_Cancer_Data_Eliminating_Mitoses_SECS_ClumpThickness.xlsx')

d1 = data1.ix[:,0:9]
t1 = data1.ix[:,9:]

d2 = data2.ix[:,0:8]
t2 = data2.ix[:,8:]

d3 = data3.ix[:,0:7]
t3 = data3.ix[:,7:]

d4 = data4.ix[:,0:6]
t4 = data4.ix[:,6:]

# fit a Decision Tree Classifier model to the data
model1 = SVC(probability=True)
x1=model1.fit(d1, t1)

model2 = SVC(probability=True)
x2=model2.fit(d2, t2)

model3 = SVC(probability=True)
x3=model3.fit(d3, t3)

model4 = SVC(probability=True)
x4=model4.fit(d4, t4)

# make predictions
expected=[]
for i in t1['Class']:
    expected.append(i)
    
predicted1 = model1.predict(d1)
predicted2 = model2.predict(d2)
predicted3 = model3.predict(d3)
predicted4 = model4.predict(d4)

# summarize the fit of Decision Tree model
print("***************Without Eliminating Any Feature***************")
print(metrics.classification_report(expected, predicted1))
print("\nConfusion Matrix=\n",metrics.confusion_matrix(expected, predicted1))
print("\nAccuracy=",model1.score(d1,t1))
print("\nKappa Score=",cohen_kappa_score(expected, predicted1))
# summarize the fit of Decision Tree model
print("***************Eliminating Mitoses Feature***************")
print(metrics.classification_report(expected, predicted2))
print("\nConfusion Matrix=\n",metrics.confusion_matrix(expected, predicted2))
print("\nAccuracy=",model2.score(d2,t2))
print("\nKappa Score=",cohen_kappa_score(expected, predicted2))
# summarize the fit of Decision Tree model
print("***************Eliminating Mitoses and Single Epithelial Cell Size***************")
print(metrics.classification_report(expected, predicted3))
print("\nConfusion Matrix=\n",metrics.confusion_matrix(expected, predicted3))
print("\nAccuracy=",model3.score(d3,t3))
print("\nKappa Score=",cohen_kappa_score(expected, predicted3))
# summarize the fit of Decision Tree model
print("***************Eliminating Mitoses and Single Epithelial Cell Size***************")
print(metrics.classification_report(expected, predicted3))
print("\nConfusion Matrix=\n",metrics.confusion_matrix(expected, predicted4))
print("\nAccuracy=",model4.score(d4,t4))
print("\nKappa Score=",cohen_kappa_score(expected, predicted4))

predict_probe1=model1.predict_proba(d1)
predict_probe1=predict_probe1[:,1]

predict_probe2=model2.predict_proba(d2)
predict_probe2=predict_probe2[:,1]

predict_probe3=model3.predict_proba(d3)
predict_probe3=predict_probe3[:,1]

predict_probe4=model4.predict_proba(d4)
predict_probe4=predict_probe4[:,1]

#Finding Area underROC, FPR, TPR

dc_auc1=roc_auc_score(t1,predict_probe1)
fpr1,tpr1,_=roc_curve(t1,predict_probe1)

dc_auc2=roc_auc_score(t2,predict_probe2)
fpr2,tpr2,_=roc_curve(t2,predict_probe2)

dc_auc3=roc_auc_score(t3,predict_probe3)
fpr3,tpr3,_=roc_curve(t3,predict_probe3)


dc_auc4=roc_auc_score(t4,predict_probe4)
fpr4,tpr4,_=roc_curve(t4,predict_probe4)

#Generating ROC
plt.rcParams["font.family"] = "Times New Roman"
plt.plot(fpr1,tpr1,linestyle="--",color="black",label='Without eliminating any feature\n(AUROC =%0.3f)'% dc_auc1)
plt.plot(fpr2,tpr2,linestyle="--",color="red",label='Eliminating Mitoses\n(AUROC =%0.3f)'% dc_auc2)
plt.plot(fpr3,tpr3,linestyle="--",color="blue",label='Eliminating Mitoses \nand Single Epithelial Cell Size\n(AUROC =%0.3f)'% dc_auc3)
plt.plot(fpr4,tpr4,linestyle="--",color="blue",label='Eliminating Mitoses, \nSingle Epithelial Cell Size\n and Clump Thickness\n(AUROC =%0.3f)'% dc_auc4)
plt.title('ROC plot for Extra Tree Classifier',fontsize=12)
plt.xlabel('False Positive Rate',fontsize=12)
plt.ylabel('True Positive Rate',fontsize=12)
plt.legend(prop={'size': 12})
plt.show()


