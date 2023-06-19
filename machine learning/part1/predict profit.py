#!/usr/bin/env python
# coding: utf-8

# In[1]:


import mysql.connector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lazypredict
from lazypredict.Supervised import LazyRegressor
from wordcloud import WordCloud
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')


# # import dataset

# In[2]:


mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="bvncxz62",
    database="supermarket"
)


# In[3]:


mydb


# In[4]:


mycursor = mydb.cursor()

# Execute SQL query to retrieve table names
mycursor.execute("SHOW TABLES")

# Fetch all table names
tables = mycursor.fetchall()

# Print the table names
for table in tables:
    print(table[0])


# In[ ]:





# In[5]:


join_query = """
SELECT *
FROM instruction
JOIN order_detail ON instruction.`Order ID` = order_detail.`Order ID`
JOIN product ON product.`Product ID` = order_detail.`Product ID`
JOIN shipping ON shipping.`Order ID` = instruction.`Order ID`;

"""


# In[6]:


mycursor.execute(join_query)
joined_data = mycursor.fetchall()


# In[7]:


column_names = [i[0] for i in mycursor.description]

hf = pd.DataFrame(joined_data, columns=column_names)

mydb.close()


# # .

# In[8]:


hf


# In[9]:


hf.columns


# In[10]:


hf.isna().sum()


# In[11]:


hf.columns


# In[12]:


df=hf[['Order ID','Order Priority','Order Date','Market','Sales','Quantity','Discount','Shipping Cost','Product Name','Category','Sub-Category','Ship Mode','Profit']]


# In[13]:


df


# In[14]:


df.columns


# In[15]:


duplicate_columns = df.columns[df.columns.duplicated()]
df = df.loc[:, ~df.columns.duplicated()]


# In[16]:


df.head()


# In[17]:


df.shape


# In[18]:


df['Quantity'].value_counts().plot(kind='bar')
plt.show()


# In[19]:


df['Order Priority'].value_counts().plot(kind='bar')
plt.show()


# In[20]:


df['Market'].value_counts().plot(kind='bar')
plt.show()


# In[21]:


df['Category'].value_counts().plot(kind='bar')
plt.show()


# In[22]:


df['Sub-Category'].value_counts().plot(kind='bar')
plt.show()


# In[23]:


df['Ship Mode'].value_counts().plot(kind='bar')
plt.show()


# In[24]:


text_data = ' '.join(df['Product Name'].dropna().astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='Set2', collocations=False).generate(text_data)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[25]:


# df['Year'] = df['Order Date'].dt.year
# df['month'] = df['Order Date'].dt.month
df['price']=df['Sales']/df['Quantity']


# In[26]:


df=df.round(2)


# In[27]:


data=df.copy()


# In[28]:


data.head()


# In[29]:


data.drop(['Order Date','Product Name','Category'],axis=1,inplace=True)


# In[30]:


data.head()


# In[31]:


plt.figure(figsize=(16,8))
sns.boxplot(data=data, x="Sales", y="Sub-Category")
plt.show()


# In[32]:


plt.figure(figsize=(16,8))
sns.boxplot(data=data, x="price", y="Sub-Category")
plt.show()


# In[33]:


# groups = data.groupby('Sub-Category')
# filtered_data = []

# for name, group in groups:
#     Q1 = group['Profit'].quantile(0.25)
#     Q3 = group['Profit'].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
#     filtered_group = group[(group['Profit'] >= lower_bound) & (group['Profit'] <= upper_bound)]
#     filtered_data.append(filtered_group)
# data_cl = pd.concat(filtered_data)


# In[34]:


plt.boxplot(data['Profit'])


# In[35]:


Q1 = data['Profit'].quantile(0.1)
Q3 = data['Profit'].quantile(0.9)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data_cl = data[(data['Profit'] >= lower_bound) & (data['Profit'] <= upper_bound)]


# In[36]:


data_cl.shape


# In[37]:


plt.boxplot(data_cl['Sales'])


# In[38]:


data_cl.shape


# In[39]:


Q1 = data_cl['Sales'].quantile(0.1)
Q3 = data_cl['Sales'].quantile(0.9)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data_cl = data_cl[(data_cl['Sales'] >= lower_bound) & (data_cl['Sales'] <= upper_bound)]


# In[40]:


plt.figure(figsize=(16,8))
sns.boxplot(data=data, x="Profit", y="Sub-Category")
plt.show()


# In[41]:


plt.figure(figsize=(16,8))
sns.boxplot(data=data_cl, x="Profit", y="Sub-Category")
plt.show()


# In[42]:


sns.pairplot(data=data_cl)
plt.show()


# In[43]:


data_cl


# In[44]:


data_cl.shape


# # numberize

# In[45]:


data_num=data_cl.copy()


# In[46]:


from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
data_num['Market']=LE.fit_transform(data_num['Market'])
data_num['Sub-Category']=LE.fit_transform(data_num['Sub-Category'])
data_num['Ship Mode']=LE.fit_transform(data_num['Ship Mode'])
# Ship_Mode = pd.get_dummies(data_num['Ship Mode'])
data_num['Order Priority']=LE.fit_transform(data_num['Order Priority'])


# In[47]:


sns.pairplot(data=data_num)


# In[ ]:





# # .

# In[50]:


data_class=data_num.drop(['Order ID','Quantity'],axis=1)
# data_class=data_class[['Sales','Sub-Category','Shipping Cost','price','S_M','D_M','Discount','Profit']]


# In[51]:


fig,ax=plt.subplots(figsize=(20,10))
corrMatrix = data_class.corr()
sns.heatmap(corrMatrix, annot=True,cmap="YlGnBu")
plt.show()


# In[52]:


data_class


# In[53]:


train_x,test_x=train_test_split(data_class,test_size=0.3)
valid_x,test_x=train_test_split(test_x,test_size=0.5)


# In[54]:


# train datas
train_x['mean_cat'] = train_x.groupby('Sub-Category')['price'].transform('mean')
train_x['P_M']=train_x['price']/train_x['mean_cat']
train_x['sales_mean_cat'] = train_x.groupby('Sub-Category')['Sales'].transform('mean')
train_x['S_M']=data['Sales']/train_x['sales_mean_cat']
train_x['dis_mean_cat'] = train_x.groupby('Sub-Category')['Discount'].transform('mean')
train_x['D_M']=train_x['Discount']/train_x['dis_mean_cat']

# validation datas
valid_x['mean_cat'] = valid_x.groupby('Sub-Category')['price'].transform('mean')
valid_x['P_M']=valid_x['price']/valid_x['mean_cat']
valid_x['sales_mean_cat'] = valid_x.groupby('Sub-Category')['Sales'].transform('mean')
valid_x['S_M']=valid_x['Sales']/valid_x['sales_mean_cat']
valid_x['dis_mean_cat'] = valid_x.groupby('Sub-Category')['Discount'].transform('mean')
valid_x['D_M']=valid_x['Discount']/valid_x['dis_mean_cat']

# test datas
test_x['mean_cat'] = test_x.groupby('Sub-Category')['price'].transform('mean')
test_x['P_M']=test_x['price']/test_x['mean_cat']
test_x['sales_mean_cat'] = test_x.groupby('Sub-Category')['Sales'].transform('mean')
test_x['S_M']=test_x['Sales']/test_x['sales_mean_cat']
test_x['dis_mean_cat'] = test_x.groupby('Sub-Category')['Discount'].transform('mean')
test_x['D_M']=test_x['Discount']/test_x['dis_mean_cat']


# In[55]:


train_x.drop(['Discount','sales_mean_cat','mean_cat','dis_mean_cat'],axis=1,inplace=True)
valid_x.drop(['Discount','sales_mean_cat','mean_cat','dis_mean_cat'],axis=1,inplace=True)
test_x.drop(['Discount','sales_mean_cat','mean_cat','dis_mean_cat'],axis=1,inplace=True)


# In[56]:


train_x


# In[57]:


MMS=MinMaxScaler()
X_train=pd.DataFrame(MMS.fit_transform(train_x),columns = train_x.columns)
X_valid=pd.DataFrame(MMS.fit_transform(valid_x),columns = valid_x.columns)
X_test=pd.DataFrame(MMS.fit_transform(test_x),columns = test_x.columns)
y_train=X_train['Profit']
y_valid=X_valid['Profit']
y_test=X_test['Profit']
X_train.drop(['Profit'],axis=1,inplace=True)
X_test.drop(['Profit'],axis=1,inplace=True)
X_valid.drop(['Profit'],axis=1,inplace=True)


# In[58]:


selected_models = ['Lasso','GaussianMixture', 'Ridge', 'ElasticNet', 'RandomForestRegressor', 'DecisionTreeRegressor',
                   'SVR', 'KNeighborsRegressor', 'AdaBoostRegressor', 'GradientBoostingRegressor',
                   'ExtraTreesRegressor', 'XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor']

# Assuming you have X_train, X_test, y_train, and y_test datasets

# Split the data into training and test sets

# Create and fit the LazyRegressor with the selected models
regressors = [reg for reg in lazypredict.Supervised.REGRESSORS if reg[0] in selected_models]
reg = LazyRegressor(verbose=0, regressors=regressors, custom_metric=r2_score)
models, predictions = reg.fit(X_train, X_valid, y_train, y_valid)


# In[59]:


predictions


# In[60]:


selected_models = ['Lasso','GaussianMixture', 'Ridge', 'ElasticNet', 'RandomForestRegressor', 'DecisionTreeRegressor',
                   'SVR', 'KNeighborsRegressor', 'AdaBoostRegressor', 'GradientBoostingRegressor',
                   'ExtraTreesRegressor', 'XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor']

# Assuming you have X_train, X_test, y_train, and y_test datasets

# Split the data into training and test sets

# Create and fit the LazyRegressor with the selected models
regressors = [reg for reg in lazypredict.Supervised.REGRESSORS if reg[0] in selected_models]
reg = LazyRegressor(verbose=0, regressors=regressors, custom_metric=r2_score)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)


# In[61]:


predictions


# In[ ]:





# In[62]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


# In[63]:


model = LinearRegression()
model.fit(X_train,y_train)
pred=model.predict(X_valid)


# In[64]:


from sklearn.metrics import  r2_score
r2_score(y_valid,pred)


# In[ ]:





# In[ ]:





# In[65]:


DTR = DecisionTreeRegressor()
DTR.fit(X_train,y_train)
pred_5=DTR.predict(X_valid)


# In[66]:


r2_score(y_valid,pred_5)


# In[ ]:





# In[ ]:





# In[69]:


import nbconvert


# In[71]:


jupyter nbconvert --to script 'all the categories-Copy2.ipynb'


# In[ ]:




