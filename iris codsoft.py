#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install numpy')
get_ipython().system('pip install seaborn')
get_ipython().system('pip install matplot')
get_ipython().system('pip install -U scikit-learn')
get_ipython().system('pip install plotly==5.7.0')


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[3]:


from warnings import filterwarnings
filterwarnings(action='ignore')


# In[4]:


iris=pd.read_csv(r"C:\Users\DELL 5502\Downloads\iris.csv")
print(iris)


# In[8]:


import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Define the full path to the image file in the Downloads directory
img_path = r"C:\Users\DELL 5502\Downloads\iris_types.jpg"

# Read the image using mpimg.imread and store it in the variable img
img = mpimg.imread(img_path)

# Display the image using Matplotlib
plt.imshow(img)
plt.axis('off')  # Turn off axis labels and ticks
plt.show()
A


# In[9]:



print(iris.shape)


# In[10]:



print(iris.describe())


# In[11]:


print(iris.isna().sum())
print(iris.describe())


# In[12]:


iris.head()


# In[13]:


iris.head(150)


# In[14]:


iris.tail(100)


# In[15]:


n = len(iris[iris['species'] == 'versicolor'])
print("No of Versicolor in Dataset:",n)


# In[16]:



n1 = len(iris[iris['species'] == 'virginica'])
print("No of Virginica in Dataset:",n1)


# In[17]:



n2 = len(iris[iris['species'] == 'setosa'])
print("No of Setosa in Dataset:",n2)


# In[18]:


iris.hist()
plt.show()


# In[21]:


species_counts = iris['species'].value_counts()
plt.bar(species_counts.index, species_counts.values)
plt.xlabel('Species')
plt.ylabel('Count')
plt.title('Bar Plot of Species Count')
plt.show()


# In[22]:


plt.pie(species_counts, labels=species_counts.index, autopct='%1.1f%%')
plt.title('Pie Chart of Species Distribution')
plt.show()


# In[23]:


plt.scatter(iris['petal_length'], iris['petal_width'])
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Scatter Plot of Petal Length vs Petal Width')
plt.show()


# In[24]:


plt.plot(iris['sepal_length'], label='Sepal Length')
plt.plot(iris['sepal_width'], label='Sepal Width')
plt.xlabel('Index')
plt.ylabel('Length/Width')
plt.title('Line Plot of Sepal Length and Width')
plt.legend()
plt.show()


# In[25]:


sns.boxplot(x='species', y='petal_width', data=iris)
plt.title('Box Plot of Petal Width by Species')
plt.show()


# In[26]:


corr_mat = iris.corr()
print(corr_mat)


# In[27]:



from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier


# In[28]:



train, test = train_test_split(iris, test_size = 0.25)
print(train.shape)
print(test.shape)


# In[30]:


train_X = train[['sepal_length', 'sepal_width', 'petal_length',
                 'petal_width']]
train_y = train.species

test_X = test[['sepal_length', 'sepal_width', 'petal_length',
                 'petal_width']]
test_y = test.species


# In[31]:


train_X.head()


# In[32]:



test_y.head()


# In[34]:



#Using LogisticRegression
model = LogisticRegression()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print('Accuracy:',metrics.accuracy_score(prediction,test_y))


# In[35]:


#Confusion matrix
from sklearn.metrics import confusion_matrix,classification_report
confusion_mat = confusion_matrix(test_y,prediction)
print("Confusion matrix: \n",confusion_mat)
print(classification_report(test_y,prediction))


# In[36]:


results = pd.DataFrame({
    'Model': ['Logistic Regression','Support Vector Machines', 'Naive Bayes','KNN' ,'Decision Tree'],
    'Score': [0.947,0.947,0.947,0.947,0.921]})

result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(9)


# In[ ]:




