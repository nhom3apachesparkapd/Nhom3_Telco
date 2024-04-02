# Nhom3_Telco
# Danh sách nhóm
1. Hoàng Thu Hương
2. Lê Lưu Thúy Hằng
3. Lê Thị Quí
4. Hà Văn Hoàng
5. Hoàng Lan Anh
6. Lê Trọng Quý
7. Nguyễn Thanh Hải
   
# import các thư viện liên quan
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

#cột dữ liệu với các services yes
data["Tong dich vu"]= (data[['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']]== 'Yes').sum(axis=1)

#tạo ra trường dữ liệu mới với những khách hàng ko được cung cấp
data['no_itn'] = 1
data.loc[(data['OnlineSecurity'] != 'No internet service') & 
         (data['OnlineBackup'] != 'No internet service') & 
         (data['DeviceProtection'] != 'No internet service') & 
         (data['TechSupport'] != 'No internet service') & 
         (data['StreamingTV'] != 'No internet service') & 
         (data['StreamingMovies'] != 'No internet service'), 'no_itn'] = 0


#biểu đồ cột đếm churn
trace = go.Bar(
        x = (data['Churn'].value_counts().values.tolist()), 
        y = ['Churn : no', 'Churn : yes'], 
        orientation = 'h', opacity = 0.8, 
        text=data['Churn'].value_counts().values.tolist(), 
        textfont=dict(size=15),
        textposition = 'auto',
        marker=dict(
        color=['green','red'],
        line=dict(color='#000000',width=1.5)
        ))

layout = dict(title =  'Biểu đồ cột đếm khách số khách hàng churn',
                        autosize = False,
                        height  = 550,
                        width   = 850)
                    
fig = dict(data = [trace], layout=layout)
py.iplot(fig)



trace = go.Pie(labels = ['Churn : no', 'Churn : yes'], values = data['Churn'].value_counts(), 
               textfont=dict(size=15), opacity = 0.8,
               marker=dict(colors=['green','red'], 
                           line=dict(color='#000000', width=1.5)))


layout = dict(title =  'Biểu đồ tròn thể hiện phần trăm churn',
                        autosize = False,
                        height  = 550,
                        width   = 850)
           
fig = dict(data = [trace], layout=layout)
py.iplot(fig)

#biểu đô phân phối 
def bieudo_hist(var_select, bin_size) : 
    tmp1 = churn[var_select]
    tmp2 = no_churn[var_select]
    hist_data = [tmp1, tmp2]
    group_labels = ['Churn : yes', 'Churn : no']
    colors = ['red', 'green']
    fig = ff.create_distplot(hist_data, group_labels, colors = colors, show_hist = True, curve_type='kde', bin_size = bin_size)
    fig['layout'].update(title = var_select, autosize = False,
                        height  = 550,
                        width   = 850)
    py.iplot(fig, filename = 'Density plot')

bieudo_hist('tenure', False)
bieudo_hist('MonthlyCharges', False)
bieudo_hist('TotalCharges', False)

palette ={0 : 'green', 1 : 'red'}
edgecolor = 'blue'
fig = plt.figure(figsize=(19,8))
alpha = 0.8

plt.subplot(131)
ax1 = sns.scatterplot(x = data['TotalCharges'], y = data['tenure'], hue = "Churn",
                    data = data, palette = palette, edgecolor=edgecolor, alpha = alpha)
plt.title('tổng thanh toán và thời gian ở lại')

plt.subplot(132)
ax2 = sns.scatterplot(x = data['TotalCharges'], y = data['MonthlyCharges'], hue = "Churn",
                    data = data, palette =palette, edgecolor=edgecolor, alpha = alpha)
plt.title('tổng thanh toán so với thanh toán hàng tháng')

plt.subplot(133)
ax2 = sns.scatterplot(x = data['MonthlyCharges'], y = data['tenure'], hue = "Churn",
                    data = data, palette =palette, edgecolor=edgecolor, alpha = alpha)
plt.title('thanh toán hàng tháng với thời gian ở lại')

fig.suptitle('biểu đồ scatter', fontsize = 20)
plt.savefig('1')
plt.show()

#biểu đồ nhiệt thể hiện sự tương quang giữa 3 biến
df_quant = data.select_dtypes(exclude=[object])
df_quant.head()
corr_quant = df_quant.corr()

fig, ax = plt.subplots(figsize=(15, 10))
ax = sns.heatmap(corr_quant, annot=True, cmap = 'viridis', linewidths = .1, linecolor = 'blue', fmt=".2f")
ax.invert_yaxis()
ax.set_title("Mức độ tương quan")
plt.show()

#hàm biểu đồ cột
def barplot(var_select, x_no_numeric) :
    tmp1 = data[(data['Churn'] != 0)]
    tmp2 = data[(data['Churn'] == 0)]
    tmp3 = pd.DataFrame(pd.crosstab(data[var_select],data['Churn']), )
    tmp3['Attr%'] = tmp3[1] / (tmp3[1] + tmp3[0]) * 100
    if x_no_numeric == True  : 
        tmp3 = tmp3.sort_values(1, ascending = False)
    trace1 = go.Bar(
        x=tmp1[var_select].value_counts().keys().tolist(),
        y=tmp1[var_select].value_counts().values.tolist(),
        text=tmp1[var_select].value_counts().values.tolist(),
        textposition = 'auto',
        name='Churn : yes',opacity = 0.8, marker=dict(
        color='red',
        line=dict(color='#000000',width=1)))
    trace2 = go.Bar(
        x=tmp2[var_select].value_counts().keys().tolist(),
        y=tmp2[var_select].value_counts().values.tolist(),
        text=tmp2[var_select].value_counts().values.tolist(),
        textposition = 'auto',
        name='Churn : no', opacity = 0.8, marker=dict(
        color='green',
        line=dict(color='#000000',width=1)))
    trace3 =  go.Scatter(   
        x=tmp3.index,
        y=tmp3['Attr%'],
        yaxis = 'y2',
        name='% Churn', opacity = 0.6, marker=dict(
        color='black',
        line=dict(color='#000000',width=0.5
        )))
    layout = dict(title =  str(var_select),  autosize = False,
                        height  = 550,
                        width   = 850,
              xaxis=dict(), 
              yaxis=dict(title= 'Count'), 
              yaxis2=dict(range= [-0, 75], 
                          overlaying= 'y', 
                          anchor= 'x', 
                          side= 'right',
                          zeroline=False,
                          showgrid= False, 
                          title= '% Churn'
                         ))
    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    py.iplot(fig)

barplot('gender', True)
barplot('SeniorCitizen', True)
barplot('Dependents', True)
barplot('PhoneService', True)
barplot('MultipleLines', True)
barplot('InternetService', True)
barplot('OnlineSecurity', True)
barplot('Partner', True)
barplot('OnlineBackup', True)
barplot('DeviceProtection', True)
barplot('TechSupport', True)
barplot('StreamingTV', True)
barplot('StreamingMovies', True)
barplot('Contract', True)
barplot('PaperlessBilling', True)
barplot('PaymentMethod', True)



#hàm biểu đô cột
def plot_distribution(feature1,feature2, df): 
    plt.figure(figsize=(18,5))
    plt.subplot(121)
    s = sns.countplot(x = feature1, hue='Churn', data = df, 
                      palette = {0 : 'green', 1 :'red'}, alpha = 0.8, 
                      linewidth = 0.4, edgecolor='blue') 
    s.set_title(feature1)
    for p in s.patches:
        s.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+30))
    plt.subplot(122)
    s = sns.countplot(x = feature2, hue='Churn', data = df, 
                      palette = {0 : 'green', 1 :'red'}, alpha = 0.8, 
                      linewidth = 0.4, edgecolor='blue') 
    s.set_title(feature2)
    for p in s.patches:
        s.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.15, p.get_height()+30))
    plt.show()

plot_distribution('SeniorCitizen', 'gender', data)
plot_distribution('Partner', 'Dependents', data)
