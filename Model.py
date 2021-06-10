import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.decomposition import FactorAnalysis
from sklearn.utils import resample

import matplotlib.pyplot as plt
import seaborn as sns

#importing data set
df=pd.read_csv("bank-additional-full.csv",sep=';')
df.drop('day_of_week', axis=1, inplace=True)
df.drop('contact', axis=1, inplace=True)
df.drop('month', axis=1, inplace=True)

#Converting highly correlated features into one (for numerical features)
fact=FactorAnalysis(n_components=1)
df['new_factor']=fact.fit_transform(df[['emp.var.rate', 'cons.price.idx','euribor3m','nr.employed']])
df.drop(['emp.var.rate', 'cons.price.idx','euribor3m','nr.employed'], axis=1, inplace=True)

#Dropping unknown values from data set
df.drop(df[df['job'] == 'unknown' ].index , inplace=True)
df.drop(df[df['marital'] == 'unknown' ].index , inplace=True)
df.drop(df[df['education'] == 'unknown' ].index , inplace=True)
df.drop(df[df['default'] == 'unknown' ].index , inplace=True)
df.drop(df[df['housing'] == 'unknown' ].index , inplace=True)
df.drop(df[df['loan'] == 'unknown' ].index , inplace=True)

#Encoding categorical features
df['poutcome']=  df['poutcome'].map({'nonexistent':1, 'failure':2, 'success':3})
df['housing'] = df['housing'].map({'no':1, 'yes':2})
df['loan'] = df['loan'].map({'no':1, 'yes':2})
df['default'] = df['default'].map({'no':1, 'yes':2})
df['job'] = df['job'].map({'housemaid':1, 'services':2, 'admin.':3, 'blue-collar':4, 'technician':5,
       'retired':6, 'management':7, 'unemployed':8, 'self-employed':9,
       'entrepreneur':10, 'student':11})
df["marital"] = df["marital"].astype('category')
df["marital"] = df["marital"].cat.codes
df["education"] = df["education"].astype('category')
df["education"] = df["education"].cat.codes
df['y'] = df['y'].map({'no':0, 'yes':1})

#Changing some numerical features into categorical
d = {range(0, 25): 3, range(25, 42): 1, range(42, 58): 2,range(58, 90): 4}
df['age'] = df['age'].apply(lambda x: next((v for k, v in d.items() if x in k), 0))
d = {range(1, 10): 1, range(10, 20): 2, range(20, 1000): 0}
df['pdays'] = df['pdays'].apply(lambda x: next((v for k, v in d.items() if x in k), 0))
d = {range(1, 60): 1, range(60, 200): 2, range(200, 600): 3, range(600, 1000): 4,range(1000, 5000): 5}
df['duration'] = df['duration'].apply(lambda x: next((v for k, v in d.items() if x in k), 0))

#Converting highly correlated features into one (for categorical features)
fact=FactorAnalysis(n_components=1)
df['newp_factor']=fact.fit_transform(df[['pdays', 'previous','poutcome']])
df.drop(['pdays', 'previous','poutcome'], axis=1, inplace=True)

#Upsampling positives
def upsample(df):
    df_M = df[df.y == 0]
    df_m = df[df.y == 1]
    df_m_upsample = resample(df_m,
                            replace = True,
                            n_samples=df_M.shape[0],
                            random_state=2)
    df_up = pd.concat([df_M,df_m_upsample],axis=0)
    y = df_up.y
    X = df_up.drop('y',axis=1)
    return X,y

X,y=upsample(df)

#Splitting data into training and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

#DecisionTree
DT = DecisionTreeClassifier(criterion="entropy",max_depth=5,max_leaf_nodes=20)
DT.fit(X_train,y_train)

#RandomForest
RF = RandomForestClassifier(n_estimators=100,n_jobs=-1, max_depth=8)
RF.fit(X_train,y_train.values.ravel())

#NaiveBayes
NB = GaussianNB(var_smoothing=6.579332246575682e-07)
NB.fit(X_train,y_train.values.ravel())

#Results
models = {'Decision Tree':DT,'Random Forest':RF,'Naive Bayes':NB}
fig, axes = plt.subplots(1, 3, sharex=True, figsize=(16,4))
for i in range(len(models.keys())):
    name = list(models.keys())[i]
    model = models[name]
    y_pred = model.predict(X_test)
    Accuracy = metrics.accuracy_score(y_test, y_pred)*100
    Recall = metrics.recall_score(y_test, y_pred)*100
    Precision = metrics.precision_score(y_test, y_pred)*100
    x = ['Recall','Precision']
    y = [Recall,Precision]
    sns.barplot(ax=axes[i], x=x, y=y)
    axes[i].set_title(name+'\n Accuracy: '+str(round(Accuracy,2)))
    axes[i].set(ylim=(0,100))
    for j in range(2):
        axes[i].text(j,y[j],round(y[j],2), color='black', ha="center")
