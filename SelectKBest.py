# Feature Selection with Univariate Statistical Tests
import pandas as pd
import numpy as np
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# load data
data=pd.read_excel('Breast_Cancer_Data.xlsx')

X = data.ix[:,0:9]

Y = data.ix[:,9:]

# feature extraction
test = SelectKBest(score_func=f_classif, k=8)
fit = test.fit(X, Y)

# summarize scores
set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)

# summarize selected features
print(features[0:5,:])
