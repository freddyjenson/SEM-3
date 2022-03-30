import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.tree import plot_tree
df = sns.load_dataset('iris')
print(df.head())
print(df.info())
df.isnull().any()
print(df.shape)
sns.pairplot(data=df,hue='species')
plt.savefig("decison_tree.png")
#correlation matrix
sns.heatmap(df.corr())
plt.savefig("one.png")
target=df['species']
df1=df.copy()
df1=df1.drop('species',axis=1)
print(df1.shape)
print(df1.head())
#defining the attribute
x=df1;
print(target)
#label encoding
le=LabelEncoder()
target=le.fit_transform(target)
print(target)
y=target

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print('Training split input- ',x_train.shape)
print('testing split input- ',x_test.shape)
#Defing the Decision tree algorithm
dtree=DecisionTreeClassifier()
dtree.fit(x_train,y_train)
y_pred=dtree.predict(x_test)
print('Classification Report - \n',classification_report(y_test,y_pred))
cm=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(data=cm,linewidth=5,annot=True,square=True,cmap="Blues")
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
all_sample_title = 'Accuracy Score : {0}'.format(dtree.score(x_test,y_test))
plt.title(all_sample_title,size= 15)
plt.savefig("2.png")
#visualizong the graph without the use of graphics
plt.figure(figsize=(20,20))
dec_tre=plot_tree(decision_tree=dtree,feature_names=df1.columns,class_names=["satosa","vercicolor","venginica"],filled=True,precision=4,rounded=True)
plt.savefig("3.png")
