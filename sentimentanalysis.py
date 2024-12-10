import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import re
import streamlit as sl

ps = PorterStemmer()
sns.set()


data = pd.read_csv("Updated_Restaurant_Reviews_Manual5000-.csv")
print(data.info())  


data['Liked'] = data['Liked'].fillna(data['Liked'].mode()[0])  
y = data['Liked'].values  



print(data['Liked'].isnull().sum())  

corpus = []
for i in range(len(data)):
    
    review_text = str(data['Review'][i]) if isinstance(data['Review'][i], float) else data['Review'][i]
    
    # Preprocessing
    review_text = re.sub('[^a-zA-Z]', " ", review_text)  
    review_text = review_text.lower() 
    review_text = review_text.split()  
    review_text = [word for word in review_text if word not in stopwords.words('english')]  
    review_text = ' '.join(review_text)  
    corpus.append(review_text)

cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

LR_model = LogisticRegression()
LR_model.fit(X_train, y_train)


y_pred = LR_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


sl.title('Welcome to Sentiment Analysis App for TasteLens AI Team')
sl.markdown('---')

form = sl.form('sentiment')
text = form.text_input('Please enter your tweet or comment here:')
sl.markdown('---')

if form.form_submit_button('Check'):
    text = re.sub('[^a-zA-Z]', " ", text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stopwords.words('english')]
    text = ' '.join(text)
    prediction = LR_model.predict(cv.transform([text]).toarray())
    
    if prediction[0] == 1:
        sl.write("The tweet or comment you entered is positive.")
    else:
        sl.write("The tweet or comment you entered is negative.")
