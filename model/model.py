import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer #Term Frequency - Inverse Document Frequency
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.metrics import precision_score, recall_score

# Read data from CSV file 
file_path = 'modelset.csv'  
df = pd.read_csv(file_path)

# Features and labels
X = df[['Result', 'Preferences']]
y = df['Preferences']

# Encoding preferences using LabelEncoder
le = preprocessing.LabelEncoder()
y_encoded = le.fit_transform(y)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# TF-IDF Vectorizer for text data
tfidf_vectorizer = TfidfVectorizer()

# Fitting the TF-IDF vectorizer on training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train['Preferences'])

# Creating a pipeline with the fitted TF-IDF vectorizer and the classifier
model = make_pipeline(tfidf_vectorizer, MultinomialNB())
model.fit(X_train['Preferences'], y_train)

# Prediction on the test set
X_test_tfidf = tfidf_vectorizer.transform(X_test['Preferences'])
y_pred = model.predict(X_test['Preferences'])

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Confusion Matrix and Classification Report
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]

    # Fitting the TF-IDF vectorizer on training data in each fold
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train['Preferences'])

    # Creating a pipeline with the fitted TF-IDF vectorizer and the classifier
    model = make_pipeline(tfidf_vectorizer, MultinomialNB())
    model.fit(X_train['Preferences'], y_train)

    # Prediction on the test set
    X_test_tfidf = tfidf_vectorizer.transform(X_test['Preferences'])
    y_pred_fold = model.predict(X_test['Preferences'])

    accuracy = accuracy_score(y_test, y_pred_fold)
    print(f'Fold Accuracy: {accuracy:.2f}')

# Real-time Feedback and Analysis (Assuming continuous streaming data)
max_iterations = 10  # Set a maximum number of iterations
iteration_count = 0

while iteration_count < max_iterations:
    # Collect real-time data (replace this with your data collection process)
    new_data = {'Result': 88, 'Preferences': 'coding'}  # Example real-time data

    # Preprocess the new data
    new_X = pd.DataFrame([new_data])
    new_X_tfidf = tfidf_vectorizer.transform(new_X['Preferences'])

    # Make a prediction using predict_proba for new data
    new_prediction_proba = model.predict_proba(new_X)

    # Extract the probability for the positive class (class with label 1)
    decision_values = new_prediction_proba[:, 1]

    # Provide feedback or take action based on the decision values
    print(f'Real-time Prediction Decision Values: {decision_values}')

    # Update the model periodically with new data if needed
    # model.partial_fit(...)

    # Analyze student behavior based on predictions and other relevant data
    # ...

    iteration_count += 1

    # Add a delay to simulate real-time streaming (replace this with your streaming mechanism)
    time.sleep(190)

