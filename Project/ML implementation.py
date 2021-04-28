#!/usr/bin/env python
# coding: utf-8

# # Data Privacy Project by Emre & Yasin

# ## The Steps and Aim of the Project:

# In this project, we want to implement privacy preserved ML techniques and we want to see how these techniques can be applied and how privacy measures affect utility. That's why we will use the steps below to implement our project:
# 
# 1- Without any privacy method implementation, we will implement ML models and calculate them with metrics.
# 
# 2- We will generalize the features and calculate the metrics.
# 
# 3- We will implement K-anonymity , l-diversity and. t-closeness and see the effects in terms of privacy and utility 

# ## Step 1

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import scipy.stats as stt


# In[2]:


train = pd.read_csv('credit_train.csv')


# As can be seen in the figure most of the Null values are in 'Months since last delinquent' feature. Hence, we drop that feature.

# In[3]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[4]:


train.drop(columns = ['Months since last delinquent'], inplace = True)


# We can't remove features 'Credit Score & Annual income' so we will use these features like this

# In[5]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[6]:


train.dropna(inplace = True, )


# In[7]:


train


# In[8]:


#To see if there is null values like ? or other identations.
for i in range(len(train.iloc[0])):
    print(train.iloc[:,i].unique(), end ='\n')
    print('\n')
    


# In[9]:


#Let's see the seperation of y values
train.loc[:,'Loan Status'].value_counts()


# In[10]:


# We will use the data name as train. As the data is train data.
train.info()


# In[11]:


#y = data['Loan Status']
#y


# In[12]:


#x = data.drop(columns = 'Loan Status')
#x


# In[13]:


train.head(2)


# train
# 
# cust_id = pd.get_dummies(train['Customer ID'],drop_first=True)
# embark = pd.get_dummies(train['Embarked'],drop_first=True)
# 
# train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
# 
# train = pd.concat([train,sex,embark],axis=1)

# In[14]:


categorical = set((
    'Loan ID',
    'Customer ID',
    'Loan Status',
    'Term',
    'Years in current job',
    'Home Ownership',
    'Purpose',
    
))

for name in categorical:
    if name == 'Years in Current Job':
        train[name] = str(train[name]).astype('category')
    else:
        train[name] = train[name].astype('category')


# In[15]:


train.info()


# In[16]:


# creating instance of labelencoder
labelencoder = LabelEncoder()

for name in categorical:
    train[name] = labelencoder.fit_transform(train[name])
    


# In[17]:


train


# ## Train Test Split

# In[18]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop('Loan Status',axis=1), 
                                                    train['Loan Status'], test_size=0.30, 
                                                    random_state=101)


# ## Training and Predicting

# In[19]:


from sklearn.linear_model import LogisticRegression


# In[ ]:





# In[20]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[21]:


predictions = logmodel.predict(X_test)
predictions


# ## Evaluation

# We can check precision,recall,f1-score using classification report!

# In[22]:


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# In[23]:


print(classification_report(y_test,predictions))


# In[24]:


accuricies = []


# In[25]:


accuricies.append(accuracy_score(y_test,predictions))
print(accuracy_score(y_test,predictions))


# In[26]:


accuricies


# In[27]:


#Log Regression Pipeline
def log_regression(train):
    
    X_train, X_test, y_train, y_test = train_test_split(train.drop('Loan Status',axis=1), 
                                                    train['Loan Status'], test_size=0.30, 
                                                    random_state=101)
    logmodel = LogisticRegression()
    logmodel.fit(X_train,y_train)
    predictions = logmodel.predict(X_test)
    print(classification_report(y_test,predictions))
    print(accuracy_score(y_test,predictions))


# ## Decision Tree Implementation
# 
# We'll start just by training a single decision tree.

# In[28]:


from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)


# In[ ]:





# ## Prediction and Evaluation 
# 
# Let's evaluate our decision tree.

# In[29]:


predictions = dtree.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,predictions))

print(accuracy_score(y_test,predictions))
accuricies.append(accuracy_score(y_test,predictions))


# In[30]:


def decision_tree(train):
    X_train, X_test, y_train, y_test = train_test_split(train.drop('Loan Status',axis=1), 
                                                    train['Loan Status'], test_size=0.30, 
                                                    random_state=101)
    dtree = DecisionTreeClassifier()
    dtree.fit(X_train,y_train)
    predictions = dtree.predict(X_test)
    print(classification_report(y_test,predictions))
    print(accuracy_score(y_test,predictions))
    


# ## Random Forests
# 
# Now let's compare the decision tree model to a random forest.

# In[31]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)


# In[32]:


rfc_pred = rfc.predict(X_test)


# In[33]:


print(confusion_matrix(y_test,rfc_pred))


# In[34]:


print(classification_report(y_test,rfc_pred))


# In[35]:


print(accuracy_score(y_test,rfc_pred))


# In[36]:


accuricies.append(accuracy_score(y_test,rfc_pred))


# In[37]:


accuricies


# In[38]:


def random_forest(train):
    X_train, X_test, y_train, y_test = train_test_split(train.drop('Loan Status',axis=1), 
                                                    train['Loan Status'], test_size=0.30, 
                                                    random_state=101)
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    rfc_pred = rfc.predict(X_test)
    print(classification_report(y_test,rfc_pred))
    print(accuracy_score(y_test,rfc_pred))


# ## Step 2

# At Step 2, we will first anonymize dataset and after that we will k-anonymize the dataset according to different k values and measure their effect. 

# ### Generalization on features

# ### Numeric Value Generalization
# #### 1.Current Loan Amount

# In[39]:


train.head(3)


# In[40]:


loan_amount = np.asarray(train['Current Loan Amount'])
credit_score = np.asarray(train['Credit Score'])


# In[41]:


plt.hist(train['Current Loan Amount'], bins = 4, color = 'g', )


# In[42]:


plt.hist(loan_amount>0.7 , bins = 4, color = 'g', )


# In[43]:


train[train['Current Loan Amount'] <= 600000]


# In[44]:


plt.hist(train[train['Current Loan Amount'] <= 100000]['Current Loan Amount'])


# In[45]:


trainx = train.copy()
trainx.head(2)


# In[46]:


def anonymizer(trainx, index, first_indice, second_indice):
    a = trainx[trainx.iloc[:,index]>= first_indice][trainx.iloc[:,index]<= second_indice].iloc[:,index]
    mean = int((first_indice + second_indice) / 2)
    for i in a.keys():
        trainx.iloc[:,index][i] = mean
    


# In[ ]:





# In[47]:


trainx[trainx.iloc[:,3]>= 80000][trainx.iloc[:,3]<= 95000].iloc[:,3]


# In[48]:


anonymizer(trainx, 3 , 0, 25000 )
for i in range(15):
    anonymizer(trainx, 3 , (25001 + i*5000), (25000 +(i+1)*5000)) 


# In[49]:


plt.hist(train[train['Current Loan Amount'] >= 100000][train['Current Loan Amount'] <= 200000]['Current Loan Amount'])


# In[50]:


trainx[trainx.iloc[:,3]>= 130000][trainx.iloc[:,3]<= 140000].iloc[:,3]


# In[51]:


for i in range(20):
    anonymizer(trainx, 3 , (100001 + i*5000), (100000 +(i+1)*5000))


# In[52]:


plt.hist(train[train['Current Loan Amount'] >= 200000][train['Current Loan Amount'] <= 300000]['Current Loan Amount'])


# In[53]:


for i in range(40):
    anonymizer(trainx, 3 , (200001 + i*2500), (200000 +(i+1)*2500))


# In[54]:


plt.hist(train[train['Current Loan Amount'] >=200000][train['Current Loan Amount'] <= 300000]['Current Loan Amount'])


# In[55]:


for i in range(40):
    anonymizer(trainx, 3 , (300001 + i*2500), (300000 +(i+1)*2500))


# In[56]:


plt.hist(train[train['Current Loan Amount'] >= 300000][train['Current Loan Amount'] <= 400000]['Current Loan Amount'])


# In[57]:


for i in range(40):
    anonymizer(trainx, 3 , (400001 + i*2500), (400000 +(i+1)*2500))


# In[58]:


plt.hist(train[train['Current Loan Amount'] >= 500000][train['Current Loan Amount'] <= 600000]['Current Loan Amount'])


# In[59]:


for i in range(20):
    anonymizer(trainx, 3 , (500001 + i*5000), (500000 +(i+1)*5000))


# In[60]:


plt.hist(train[train['Current Loan Amount'] >= 700000][train['Current Loan Amount'] <= 800000]['Current Loan Amount'])


# In[61]:


for i in range(20):
    anonymizer(trainx, 3 , (600001 + i*5000), (600000 +(i+1)*5000))


# In[62]:


for i in range(10):
    anonymizer(trainx, 3 , (700001 + i*10000), (700000 +(i+1)*10000))


# In[63]:


plt.hist(train[train['Current Loan Amount'] >= 800000][train['Current Loan Amount'] <= 99999999]['Current Loan Amount'])


# In[64]:


trainx['Current Loan Amount']


# In[65]:


trainx


# #### 2.Credit Score Generalization

# In[66]:


plt.hist(trainx[trainx['Credit Score'] >7500][trainx['Credit Score'] <7600]['Credit Score'])


# In[67]:


plt.hist(train[train['Credit Score'] >6]['Credit Score'])


# In[68]:


trainx[trainx['Credit Score'] >500][trainx['Credit Score'] <600]['Credit Score']


# In[69]:


anonymizer(trainx, 5 , (500), (600))


# In[70]:


for i in range(13):
    anonymizer(trainx, 5 , (601 + i*3), (601 +(i+1)*3))


# In[71]:


for i in range(5):
    anonymizer(trainx, 5 , (641 + i*2), (640 +(i+1)*2))
for i in range(5):
    anonymizer(trainx, 5 , (651 + i*2), (650 +(i+1)*2))
for i in range(5):
    anonymizer(trainx, 5 , (641 + i*2), (640 +(i+1)*2))
    
#No need to do anymore as there are enough equivalent values for credit values higher that 650.


# In[72]:


anonymizer(trainx, 5 , (5800), (6200))
for i in range(3):
    anonymizer(trainx, 5 , (6201 + i*100), (6500 +(i+1)*100))
for i in range(6):
    anonymizer(trainx, 5 , (6501 + i*50), (6500 +(i+1)*50))
for i in range(8):
    anonymizer(trainx, 5 , (6801 + i*25), (6800 +(i+1)*25))
for i in range(10):
    anonymizer(trainx, 5 , (7001 + i*10), (7000 +(i+1)*10))
for i in range(10):
    anonymizer(trainx, 5 , (7101 + i*10), (7100 +(i+1)*10))
for i in range(10):
    anonymizer(trainx, 5 , (7201 + i*10), (7200 +(i+1)*10))
for i in range(10):
    anonymizer(trainx, 5 , (7301 + i*10), (7300 +(i+1)*10))
for i in range(10):
    anonymizer(trainx, 5 , (7401 + i*10), (7400 +(i+1)*10))                
for i in range(10):
    anonymizer(trainx, 5 , (7501 + i*10), (7500 +(i+1)*10))    


# In[73]:


trainx[trainx['Credit Score'] >680][trainx['Credit Score'] <690]['Credit Score']


# In[74]:


plt.hist(trainx[trainx['Credit Score'] >600][trainx['Credit Score'] <630]['Credit Score'])


# In[75]:


trainx


# ### 3. Years in current job

# In[76]:


plt.hist(trainx['Years in current job'])


# In[77]:


anonymizer(trainx, 7, 0, 5)
anonymizer(trainx, 7, 5, 10)


# ### 4. Home Ownership

# In[78]:


plt.hist(trainx['Home Ownership'])


# In[79]:



anonymizer(trainx, 8, 0, 2)
anonymizer(trainx, 8, 2, 4)


# In[80]:


plt.hist(trainx['Home Ownership'])


# In[81]:


trainx['Home Ownership'].value_counts()


# In[82]:


trainx


# ### 5. Monthly Debt

# In[83]:



plt.hist(trainx[trainx['Monthly Debt'] >0][trainx['Monthly Debt'] <70000]['Monthly Debt'])


# In[84]:


for i in range(20):
    anonymizer(trainx, 10, (0 + i*2000) , (0 + (i+1) * 2000))
for i in range(8):
    anonymizer(trainx, 10, (40000 + i*5000) , (40000 + (i+1) * 5000))
    


# In[85]:


plt.hist(trainx[trainx['Monthly Debt'] >0][trainx['Monthly Debt'] <10000]['Monthly Debt'])


# In[86]:


trainx['Monthly Debt'].value_counts()


# ### 6. Years of Credit History

# In[87]:


trainx.head(2)


# In[88]:


plt.hist(trainx[trainx['Years of Credit History'] > 5 ][trainx['Years of Credit History']<50]['Years of Credit History'])


# In[89]:


anonymizer(trainx, 11, 0 , 5 )
for i in range(10):
    anonymizer(trainx, 11, 5 + i * 3  , 5 + (i+1) *3 )
for i in range(3):
    anonymizer(trainx, 11, 35 + i * 5  , 35 + (i+1) *5 )


# In[90]:


plt.hist(trainx[trainx['Years of Credit History'] > 5 ][trainx['Years of Credit History']<10]['Years of Credit History'])


# ### 7. Number of Open Accounts

# In[91]:


plt.hist(trainx[trainx['Number of Open Accounts'] > 0 ][trainx['Number of Open Accounts']<30]['Number of Open Accounts'])


# In[92]:


for i in range(10):
    anonymizer(trainx, 12, 0 + i *3 , 0 +(i+1)*3 )


# ### 8. Number of Credit Problems

# In[93]:


trainx['Number of Credit Problems'].value_counts()


# In[94]:


anonymizer(trainx, 13, 0, 2)
anonymizer(trainx, 13, 3, 5)
anonymizer(trainx, 13, 6, 10)


# ### 9. Current Credit Balance

# In[95]:


plt.hist(trainx[trainx['Current Credit Balance'] <900000]['Current Credit Balance'])


# In[96]:


for i in range(90):
    anonymizer(trainx, 14, 0 + 10000 *i , 0 + 10000 *(i+1) )


# In[97]:


trainx


# ### 10. Maximum Open Credit

# In[98]:


plt.hist(trainx[trainx['Maximum Open Credit'] <4000000]['Maximum Open Credit'])


# In[99]:


for i in range(125):
    anonymizer(trainx, 15, 0 + 20000 *i , 0 + 20000 *(i+1) )


# ### 11. Bankruptcies

# In[100]:


trainx['Bankruptcies']


# We don't change these because we don't need to as we are going to apply k anonymity.

# ### 12. Tax Liens

# In[101]:


trainx['Tax Liens'].value_counts()


# In[102]:


anonymizer(trainx, 17, 0, 2)
anonymizer(trainx, 17, 3, 5)
anonymizer(trainx, 17, 6, 15)


# In[103]:


trainx['Tax Liens'].unique()


# In[109]:


log_regression(trainx)


# In[110]:


random_forest(trainx)


# In[111]:


decision_tree(trainx)


# In[112]:


accuricies


# In[114]:


accuricies2 = [0.8364248123544129, 0.7846173755499957, 0.848114916745751]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[104]:


count=0 
for i in train['Current Loan Amount'].value_counts():
    if train['Current Loan Amount'].value_counts() <2 == True: 
        count += 1
count


# In[ ]:


train[train['Current Loan Amount'] < 99999998]


# In[ ]:


valuess = train[train['Current Loan Amount'] <999999999]['Current Loan Amount'].value_counts()
valuess = np.asarray(valuess)
valuess.sort()
valuess


# In[ ]:


loan_amount.sort()
loan_amount


# In[ ]:


train[train['Current Loan Amount'] > 99999998]['Loan Status'].value_counts()


# In[ ]:


loan_amount


# In[ ]:





# In[225]:


trainx.columns


# In[226]:


trainx.head(3)


# In[227]:


categorical = set((
    'Customer ID', 'Loan Status', 'Current Loan Amount', 'Term',
       'Credit Score', 'Annual Income', 'Years in current job',
       'Home Ownership', 'Purpose', 'Monthly Debt', 'Years of Credit History',
       'Number of Open Accounts', 'Number of Credit Problems',
       'Current Credit Balance', 'Maximum Open Credit', 'Bankruptcies',
       'Tax Liens'
))
categorical


# In[228]:


for name in categorical:
    trainx[name] = trainx[name].astype('category')


# In[ ]:





# In[229]:


def get_spans(df, partition, scale=None):
    """
    :param        df: the dataframe for which to calculate the spans
    :param partition: the partition for which to calculate the spans
    :param     scale: if given, the spans of each column will be divided
                      by the value in `scale` for that column
    :        returns: The spans of all columns in the partition
    """
    spans = {}
    for column in df.columns:
        if column in categorical:
            span = len(df[column][partition].unique())
        else:
            span = df[column][partition].max()-df[column][partition].min()
        if scale is not None:
            span = span/scale[column]
        spans[column] = span
    return spans


# In[230]:


full_spans = get_spans(trainx, trainx.index)


# In[231]:


def split(df, partition, column):
    """
    :param        df: The dataframe to split
    :param partition: The partition to split
    :param    column: The column along which to split
    :        returns: A tuple containing a split of the original partition
    """
    dfp = df[column][partition]
    if column in categorical:
        values = dfp.unique()
        lv = set(values[:len(values)//2])
        rv = set(values[len(values)//2:])
        return dfp.index[dfp.isin(lv)], dfp.index[dfp.isin(rv)]
    else:        
        median = dfp.median()
        dfl = dfp.index[dfp < median]
        dfr = dfp.index[dfp >= median]
        return (dfl, dfr)


# In[232]:


def is_k_anonymous(df, partition, sensitive_column, k=3):
    """
    :param               df: The dataframe on which to check the partition.
    :param        partition: The partition of the dataframe to check.
    :param sensitive_column: The name of the sensitive column
    :param                k: The desired k
    :returns               : True if the partition is valid according to our k-anonymity criteria, False otherwise.
    """
    if len(partition) < k:
        return False
    return True

def partition_dataset(df, feature_columns, sensitive_column, scale, is_valid):
    """
    :param               df: The dataframe to be partitioned.
    :param  feature_columns: A list of column names along which to partition the dataset.
    :param sensitive_column: The name of the sensitive column (to be passed on to the `is_valid` function)
    :param            scale: The column spans as generated before.
    :param         is_valid: A function that takes a dataframe and a partition and returns True if the partition is valid.
    :returns               : A list of valid partitions that cover the entire dataframe.
    """
    finished_partitions = []
    partitions = [df.index]
    while partitions:
        partition = partitions.pop(0)
        spans = get_spans(df[feature_columns], partition, scale)
        for column, span in sorted(spans.items(), key=lambda x:-x[1]):
            lp, rp = split(df, partition, column)
            if not is_valid(df, lp, sensitive_column) or not is_valid(df, rp, sensitive_column):
                continue
            partitions.extend((lp, rp))
            break
        else:
            finished_partitions.append(partition)
    return finished_partitions


# In[233]:


trainx.head(1)


# In[234]:


trainx.info()


# In[235]:


# we apply our partitioning method to two columns of our dataset, using "income" as the sensitive attribute
feature_columns = ['Home Ownership', 'Term', 'Current Loan Amount']
sensitive_column = 'Loan Status'
finished_partitions = partition_dataset(train, feature_columns, sensitive_column, full_spans, is_k_anonymous)


# In[236]:


finished_partitions


# In[240]:


len(finished_partitions)


# In[241]:


import matplotlib.pylab as pl
import matplotlib.patches as patches


# In[242]:


def build_indexes(df):
    indexes = {}
    for column in categorical:
        values = sorted(df[column].unique())
        indexes[column] = { x : y for x, y in zip(values, range(len(values)))}
    return indexes

def get_coords(df, column, partition, indexes, offset=0.1):
    if column in categorical:
        sv = df[column][partition].sort_values()
        l, r = indexes[column][sv[sv.index[0]]], indexes[column][sv[sv.index[-1]]]+1.0
    else:
        sv = df[column][partition].sort_values()
        next_value = sv[sv.index[-1]]
        larger_values = df[df[column] > next_value][column]
        if len(larger_values) > 0:
            next_value = larger_values.min()
        l = sv[sv.index[0]]
        r = next_value
    # we add some offset to make the partitions more easily visible
    l -= offset
    r += offset
    return l, r

def get_partition_rects(df, partitions, column_x, column_y, indexes, offsets=[0.1, 0.1]):
    rects = []
    for partition in partitions:
        xl, xr = get_coords(df, column_x, partition, indexes, offset=offsets[0])
        yl, yr = get_coords(df, column_y, partition, indexes, offset=offsets[1])
        rects.append(((xl, yl),(xr, yr)))
    return rects

def get_bounds(df, column, indexes, offset=1.0):
    if column in categorical:
        return 0-offset, len(indexes[column])+offset
    return df[column].min()-offset, df[column].max()+offset


# In[243]:


indexes = build_indexes(trainx)
column_x, column_y = feature_columns[:2]
rects = get_partition_rects(trainx, finished_partitions, column_x, column_y, indexes, offsets=[0.0, 0.0])

print(rects[:10])

def plot_rects(df, ax, rects, column_x, column_y, edgecolor='black', facecolor='none'):
    for (xl, yl),(xr, yr) in rects:
        ax.add_patch(patches.Rectangle((xl,yl),xr-xl,yr-yl,linewidth=1,edgecolor=edgecolor,facecolor=facecolor, alpha=0.5))
    ax.set_xlim(*get_bounds(df, column_x, indexes))
    ax.set_ylim(*get_bounds(df, column_y, indexes))
    ax.set_xlabel(column_x)
    ax.set_ylabel(column_y)

pl.figure(figsize=(20,20))
ax = pl.subplot(111)
plot_rects(trainx, ax, rects, column_x, column_y, facecolor='r')
pl.scatter(trainx[column_x], trainx[column_y])
pl.show()


# In[246]:


def agg_categorical_column(series):
    return [','.join(set(series))]

def agg_numerical_column(series):
    return [series.mean()]

def build_anonymized_dataset(df, partitions, feature_columns, sensitive_column, max_partitions=None):
    aggregations = {}
    for column in feature_columns:
        if column in categorical:
            aggregations[column] = agg_categorical_column
        else:
            aggregations[column] = agg_numerical_column
    rows = []
    for i, partition in enumerate(partitions):
        if i % 100 == 1:
            print("Finished {} partitions...".format(i))
        if max_partitions is not None and i > max_partitions:
            break
        
        grouped_columns = df.loc[partition].agg(aggregations, squeeze=False)
        sensitive_counts = df.loc[partition].groupby(sensitive_column).agg({sensitive_column : 'count'})
        values = grouped_columns.iloc[0].to_dict()
        for sensitive_value, count in sensitive_counts[sensitive_column].items():
            if count == 0:
                continue
            values.update({
                sensitive_column : sensitive_value,
                'count' : count,

            })
            rows.append(values.copy())
    return pd.DataFrame(rows)


# In[247]:


train_n = build_anonymized_dataset(trainx, finished_partitions, feature_columns, sensitive_column, max_partitions = None)


# In[181]:


print(train_n)


# In[ ]:





# In[ ]:





# In[ ]:




