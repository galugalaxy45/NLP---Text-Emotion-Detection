#!/usr/bin/env python
# coding: utf-8

# # <span style = "color:green"> Text-Based Emotion Detection</span>

# ***

# Emotion detection (ED) is a brach of sentiment analysis that deals with the extraction and analysis of emotions. The evolution of web 2.0 has put text mining and analysis at the frontiers of organizational success. It helps service provider provide tailor-made services to their customers. Numerous studies are being carried out in the area of text mining and analysis due to the ease in sourcing for data and the vast benefits its deliverable offers.

# ### Content

# There are two columns
# * Text
# * Emotion
# 
# The emotions column has various categories ranging from happiness to sadness to love and fear.

# ## Let's Begin

# ### Import necessary libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import re
import matplotlib.pyplot as plt
import nltk


# ### Read 'Emotion_final.csv' and store it in a dataframe variable

# In[3]:


data=pd.read_csv('Emotion_final.csv',encoding='ISO-8859-1')


# ### View head

# In[4]:


data.head()


# ### Print some of the texts

# In[5]:


for sentence in data['Text'].head(10):
    print(sentence)


# ### Check unique values in Emotion

# In[7]:


data['Emotion'].unique()


# ### View info of the dataset

# In[8]:


data.info()


# ### Check for null values, Remove if any

# In[10]:


data.isna().sum()


# ### Check for duplicates, Remove if any

# In[11]:


data.duplicated().sum()


# In[12]:


data.drop_duplicates(keep='first',inplace=True)


# In[13]:


data.duplicated().sum()


# ### Print some of the happy text

# In[18]:


# Print the first 10 texts for the 'happy' emotion
happy_texts = data[data['Emotion'] == 'happy']['Text'].head(10) 

for text in happy_texts:
    print(text)
    print()


# ### Print some of the sadness texts

# In[19]:


# Print the first 10 texts for the 'sadness' emotion
sadness_texts = data[data['Emotion'] == 'sadness']['Text'].head(10) 

for text in sadness_texts:
    print(text)
    print()


# ### Print some of the surpise texts

# In[21]:


# Print the first 10 texts for the 'happy' emotion
surprise_texts = data[data['Emotion'] == 'surprise']['Text'].head(10) 

for text in surprise_texts:
    print(text)
    print()


# ### Plot a countplot of Emotions

# In[25]:


sns.countplot(y='Emotion',data=data)
plt.show()


# ### Convert Emotions column to numerical values using Label encoder

# In[26]:


from sklearn.preprocessing import LabelEncoder


# In[27]:


label_encoder = LabelEncoder()


# In[28]:


data['Emotion'] = label_encoder.fit_transform(data['Emotion'])


# In[30]:


data.head()


# ### Store the classes in a list
# * use le.classes_ to fetch the classes

# In[34]:


# Access the classes
emotion_classes = label_encoder.classes_

# Display the classes
print(emotion_classes)


# ### Import WordNetLemmatizer, stopwords

# In[35]:


from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


# ### Create a function to preprocess the text (lemmatize,lower,remove stopwords, remove special characters)

# In[36]:


def preprocess(sentence):
    
    #removes all the special characters and split the sentence at spaces
    text=re.sub(r'[^0-9a-zA-Z]+',' ',sentence).split()
    
    # converts words to lowercase and removes any stopwords
    words = [x.lower() for x in text if x not in stopwords.words('english')]
    
    # Lemmatize the words
    lemma = WordNetLemmatizer()
    word = [lemma.lemmatize(word,'v') for word in words ]
    
    # convert the list of words back into a sentence
    word = ' '.join(word)
    return word


# ### Apply the function to Text in our dataframe

# In[38]:


data['Text'] =data['Text'].apply(preprocess)


# ### View some of the texts after preprocessing

# In[39]:


for i in range(10):
    print(data.iloc[i]['Text'])
    print()


# ### Convert text to vectors

# In[40]:


X=data['Text']
y=data['Emotion']


# In[41]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[42]:


vectorizer=TfidfVectorizer(ngram_range=(2,2))


# In[43]:


X=vectorizer.fit_transform(X)


# In[44]:


type(X)


# In[45]:


X.shape


# In[46]:


y.shape


# ### Split the dataset into training and Testing set

# In[47]:


from sklearn.model_selection import train_test_split


# In[48]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[50]:


X_train.shape


# In[51]:


X_test.shape


# ### Create a Random forest classifier model

# In[52]:


from sklearn.ensemble import RandomForestClassifier


# In[54]:


random_forest_model = RandomForestClassifier(n_estimators=50, random_state=42)


# ### Train the model

# In[55]:


random_forest_model.fit(X_train,y_train)


# In[56]:


from sklearn.metrics import accuracy_score, classification_report, precision_score, confusion_matrix


# ### Check the score of the model

# In[57]:


random_forest_model.score(X_train,y_train)


# ### Make predictions with X_test

# In[59]:


y_pred=random_forest_model.predict(X_test)


# ### Check the accuracy of our prediction

# In[60]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[61]:


accuracy_score(y_test,y_pred)


# ### Create confusion matrix

# In[63]:


sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,fmt='d')
plt.show()


# ### Create classification report

# In[65]:


print(classification_report(y_test,y_pred))


# ***
