# imporrt packages
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import classification_report,f1_score,precision_score,recall_score,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
# print dataset
bank = pd.read_csv("C:\\Users\\ACER\\Desktop\\scrap\\Logistic regression\\bankfull.csv",sep=';')
gh=pd.read_csv("C:\\Users\\ACER\\Desktop\\scrap\\Logistic regression\\bankfull.csv",sep=';')
bank.columns
#count , plot bar graph & convert string into unique integer vaalues
# card job
j1=list(bank["job"].unique())
j2=bank['job']
j3=[0,0,0,0,0,0,0,0,0,0,0,0]
for i in range(len(j1)):
    for j in range(len(j2)):
        if j2[j]==j1[i]:
            j3[i]=j3[i]+1
bk1 ={"j1":j1[0:12],"j3":j3[:12]}
bk1
bk_da = pd.DataFrame.from_dict(bk1)
bk_da
bk3 = j3
bk4 = j1
for i, v in enumerate(bk3):
    plt.text(i-.25, 
              v, 
              bk3[i], 
              fontsize=9, 
              color="red")
plt.bar(bk4,bk3)
plt.xticks(bk4,rotation=90)
plt.title('job')
# marital
m1=list(bank["marital"].unique())
m2=bank['marital']
m3=[0,0,0]
for i in range(len(m1)):
    for j in range(len(m2)):
        if m2[j]==m1[i]:
            m3[i]=m3[i]+1
mar1 ={"m1":m1[0:3],"m3":m3[:3]}
mar1
ma_da = pd.DataFrame.from_dict(mar1)
ma_da
ma3 = m3
ma4 = m1
for i, v in enumerate(ma3):
    plt.text(i-.25,
              v, 
              ma3[i], 
              fontsize=9, 
              color="red")
plt.bar(ma4,ma3)
plt.xticks(ma4,rotation=90)
plt.title('marital')
# education
e1=list(bank["education"].unique())
e2=bank['education']
e3=[0,0,0,0]
for i in range(len(e1)):
    for j in range(len(e2)):
        if e2[j]==e1[i]:
            e3[i]=e3[i]+1
edu1 ={"e1":e1[0:3],"e3":e3[:3]}
edu1
ed_da = pd.DataFrame.from_dict(mar1)
ed_da
ed3 = e3
ed4 = e1
for i, v in enumerate(ed3):
    plt.text(i-.25,
              v, 
              ed3[i], 
              fontsize=9, 
              color="red")
plt.bar(ed4,ed3)
plt.xticks(ed4,rotation=90)
plt.title('education')
# default
d1=list(bank["default"].unique())
d2=bank['default']
d3=[0,0]
for i in range(len(d1)):
    for j in range(len(d2)):
        if d2[j]==d1[i]:
            d3[i]=d3[i]+1
def1 ={"d1":d1[0:2],"d3":d3[:2]}
def1
de_da = pd.DataFrame.from_dict(def1)
de_da
de3 = d3
de4 = d1
for i, v in enumerate(de3):
    plt.text(i-.25,
              v, 
              de3[i], 
              fontsize=9, 
              color="red")
plt.bar(de4,de3)
plt.xticks(de4,rotation=90)
plt.title('default')
# housing
h1=list(bank["housing"].unique())
h2=bank['housing']
h3=[0,0]
for i in range(len(h1)):
    for j in range(len(h2)):
        if h2[j]==h1[i]:
            h3[i]=h3[i]+1
hou1 ={"h1":h1[0:2],"h3":h3[:2]}
hou1
ho_da = pd.DataFrame.from_dict(hou1)
ho_da
ho3 = h3
ho4 = h1
for i, v in enumerate(ho3):
    plt.text(i-.25,
              v, 
              ho3[i], 
              fontsize=9, 
              color="red")
plt.bar(ho4,ho3)
plt.xticks(ho4,rotation=90)
plt.title('housing')
# output y
out1=list(bank["y"].unique())
out2=bank['y']
out3=[0,0]
for i in range(len(out1)):
    for j in range(len(out2)):
        if out2[j]==out1[i]:
            out3[i]=h3[i]+1
ot1 ={"out1":out1[0:2],"out3":out3[:2]}
ot1
ot_da = pd.DataFrame.from_dict(ot1)
ot_da
ot3 = out3
ot4 = out1
for i, v in enumerate(ot3):
    plt.text(i-.25,
              v, 
              ot3[i], 
              fontsize=9, 
              color="red")
plt.bar(ot4,ot3)
plt.xticks(ot4,rotation=90)
plt.title('output y')
# graph plot between marital & housing
pd.crosstab(gh.housing,gh.marital).plot(kind="bar",stacked=False)
plt.xlabel('marital')
plt.ylabel('housing')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.title('marital vs housing')
# plot between job & housing
pd.crosstab(gh.job,gh.housing).plot(kind="bar",stacked=False)
plt.xlabel('job')
plt.ylabel('housing')
plt.xticks(rotation=90)
plt.grid(axis='y')
plt.title('job vs housing')
# plot between job &n marital
pd.crosstab(gh.job,gh.marital).plot(kind="bar",stacked=False)
plt.xlabel('job')
plt.ylabel('marital')
plt.xticks(rotation=90)
plt.grid(axis='y')
plt.title('job vs marital')
# plot between job & loan
pd.crosstab(gh.job,gh.loan).plot(kind="bar",stacked=False)
plt.xlabel('job')
plt.ylabel('loan')
plt.xticks(rotation=90)
plt.grid(axis='y')
plt.title('job vs loan')
# plot between married & loan
pd.crosstab(gh.marital,gh.loan).plot(kind="bar",stacked=False)
plt.xlabel('married')
plt.ylabel('loan')
plt.xticks(rotation=90)
plt.grid(axis='y')
plt.title('married vs loan')
# plot between housing & loian
pd.crosstab(gh.housing,gh.loan).plot(kind="bar",stacked=False)
plt.xlabel('housing')
plt.ylabel('loan')
plt.xticks(rotation=90)
plt.grid(axis='y')
plt.title('housing vs loan')
# drop unwanted column
bank.drop(["contact","day","month","campaign","pdays","previous","poutcome"],inplace=True,axis=1)
bank.shape
# split into train and test dataset
X = bank.iloc[:,:9]
Y = bank.iloc[:,-1].astype('int')
xtrain,xtest,ytrain,ytest = train_test_split(X,Y)
# Logistic regression
classifier = LogisticRegression()
classifier.fit(xtrain,ytrain)
y_pred = classifier.predict(xtest)   # value predict
accuracy_score(y_pred,ytest)           # calculate accuracy
f1_score(y_pred,ytest)               # calculate f1 score
precision_score(ytest, y_pred)       # calculate precision score
recall_score(ytest,y_pred)             # calculate recall score
# predict probabilities
ns_probs = [0 for _ in range(len(ytest))]
lr_probs = classifier.predict_proba(xtest)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(ytest, ns_probs)
lr_auc = roc_auc_score(ytest, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(ytest, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(ytest, lr_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
plt.xlabel('False Positive Rate')     # axis labels
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()





























