import warnings
warnings.filterwarnings("ignore")
import numpy as np
import scipy as sp
import pandas as pd

data=pd.read_excel('Breast_Cancer_Data.xlsx')

d = data.ix[:,0:9]
d_cor = d.corr()
print(pd.DataFrame(np.linalg.inv(d.corr().values), index = d_cor.index, columns=d_cor.columns))
print(pd.DataFrame(np.linalg.inv(d.corr().values).diagonal()))

