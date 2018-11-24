import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
raw_data=pd.read_csv("datar.csv")
temp=set(raw_data['icon'])
print(temp)
size_mapping={label:index for index, label in enumerate(temp)}
raw_data['icon']=raw_data['icon'].map(size_mapping)
sigma=0.2
mu=0
wanttochange=0.607*raw_data['temperature'].values+10.092+np.random.normal(mu,sigma,raw_data.shape[0])
raw_data['wantToChange']=wanttochange
raw_data[raw_data.isnull().values==True]#查看缺失值
raw_data['pressure']=raw_data['pressure'].fillna(raw_data['pressure'].mean())
raw_data['windBearing']=raw_data['windBearing'].fillna(raw_data['windBearing'].mean())
raw_data['windSpeed']=raw_data['windSpeed'].fillna(raw_data['windSpeed'].mean())
raw_data.sort_values(by="time")
raw_data.to_csv('datanoraw.csv',index=False)
sns.pairplot(raw_data,diag_kind='kde')
plt.savefig('datasetprocessingpairplot1')
raw_data
