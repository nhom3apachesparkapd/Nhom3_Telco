# Nhom3_Telco

#import các thư viện liên quan
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss
%matplotlib inline
import itertools
import lightgbm as lgbm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, confusion_matrix,  roc_curve, precision_recall_curve, accuracy_score, roc_auc_score
from datetime import datetime
import lightgbm as lgbm
import warnings
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
import warnings

import pandas as pd
data = pd.read_csv("D:/file Zalo/Telco Customer Churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")


#hiển thị dữ liệu
print("hiển thị dữ liệu: ")
display(data.head())

#thống kê dữ liệu
print("\n thống kê về dữ liệu: ")
display(data.describe())

#kích thước dữ liệu
print("\n kích thước dữ liệu: ")
display(data.shape)


#thông tin về các trường dữ liệu
print("\n thông tin các trường dữ liệu")
display(data.info())

# chuyển churn về dạng nhị phân
data.Churn.replace(to_replace = dict(Yes = 1, No = 0), inplace = True)

#ép dữ liệu sang object
col_name = ['SeniorCitizen', 'Churn']
data[col_name] = data[col_name].astype(object)

# làm sạch cột Total Charges 
data['TotalCharges'] = data['TotalCharges'].replace(" ", 0).astype('float64')

#tách churn ra làm 2 phần
churn = data[(data['Churn'] != 0)]
no_churn = data[(data['Churn'] == 0)]

data["Tong dich vu"]= (data[['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']]== 'Yes').sum(axis=1)
