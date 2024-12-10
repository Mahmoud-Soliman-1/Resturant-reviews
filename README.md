
# **🍽️ Restaurant Reviews Sentiment Analysis Project**

Welcome to the **Sentiment Analysis** project where we analyze customer reviews for a restaurant and predict whether they are **positive** or **negative**. This project leverages **Natural Language Processing (NLP)** techniques and is deployed using **Streamlit** for a user-friendly interface.

## **Project Overview**

This project analyzes a dataset of restaurant reviews. The dataset contains:

- **4181 rows**
- Two columns:
  - `Review`: The text of the customer's review.
  - `Liked`: A binary column where `0` represents a negative review, and `1` represents a positive review.

The goal of this project is to classify customer reviews as **positive** or **negative** using a **[Logistic Regression, SVM, DecisionTree, Naive Bayes, Random Forest, and KNN ]** models.

---

## **Steps Taken in the Project**

### 1. **Data Exploration & Cleaning**
- **Initial Exploration**: We explored the dataset, checking for null values and basic statistics.
- **Missing Data**: Removed rows with missing values.
- **Word Frequency**: Identified the most frequent words in the reviews, with “**food**” being the most common.

### 2. **Data Preprocessing**
- **Stop Words Removal**: Using **NLTK**, we removed common stop words like conjunctions and prepositions that don’t add much meaning (e.g., "is", "at", "the").
- **Special Characters**: Applied **Regular Expressions (RE)** to clean the text by removing punctuation, numbers, and special symbols.
- **Processing All Reviews**: We applied a loop to process the entire dataset using the steps above for each review.

### 3. **Feature Extraction**
- **CountVectorizer**: We used this technique to convert the text data into numerical values based on word frequency for further analysis.

### 4. **Modeling**
- **Data Splitting**: Split the data into training and testing sets.
- **Logistic Regression**: Trained a Logistic Regression model to predict the sentiment of the reviews.
- **Model Accuracy**: Achieved a models best accuracy is **94.3%** on the test set.
- **Confusion Matrix**: Evaluated model performance using a confusion matrix to understand the number of true positives, true negatives, false positives, and false negatives.

### 5. **Deployment**
- **Streamlit Interface**: Created a simple user interface using **Streamlit**.
  - The user enters a review, and the model predicts whether the review is positive or negative.
  
---

## **Model Performance**
With the **Logistic Regression** model, we achieved  -->> **Accuracy**: 94.3%
With the **SVM** model, we achieved  -->> **Accuracy**: 94.2%
With the **DecisionTree** model, we achieved  -->> **Accuracy**: 94.1%
With the **Naive Bayes** model, we achieved  -->> **Accuracy**: 93.9%
With the **Random Forest** model, we achieved  -->> **Accuracy**: 93.8%
With the **KNN** model, we achieved  -->> **Accuracy**: 91.1%


- **Confusion Matrix**: Provided insights into the classification performance.

---
---

## **Contributors**
- **Mahmoud Soliman**
- **Yara Mohamed**
- **Mariam Ismail**
- **Mona Abdelrazek**
- **Moaz Amen**
---

