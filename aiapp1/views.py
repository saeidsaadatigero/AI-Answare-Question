from django.shortcuts import render
from django.http import JsonResponse

# Import necessary libraries for machine learning
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset for training
train_data = pd.read_csv('Train-word.csv', sep='\t', encoding='utf-8')
val_data = pd.read_csv('Val-word.csv', sep='\t', encoding='utf-8')
test_data = pd.read_csv('Test-word.csv', sep='\t', encoding='utf-8')

# Split the dataset into training and testing sets
X_train = train_data['premise']
X_val = val_data['premise']
X_test = test_data['premise']

y_train = train_data['label']
y_val = val_data['label']
y_test = test_data['label']

# Create feature vectors using TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_val_vectorized = vectorizer.transform(X_val)
X_test_vectorized = vectorizer.transform(X_test)

# Train the machine learning model
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# Evaluate the model on the validation set
y_val_pred = model.predict(X_val_vectorized)
val_accuracy = accuracy_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred, average='weighted')
val_recall = recall_score(y_val, y_val_pred, average='weighted')
val_f1 = f1_score(y_val, y_val_pred, average='weighted')

print(f'Validation Accuracy: {val_accuracy:.2f}')
print(f'Validation Precision: {val_precision:.2f}')
print(f'Validation Recall: {val_recall:.2f}')
print(f'Validation F1-score: {val_f1:.2f}')

# Evaluate the model on the testing set
y_test_pred = model.predict(X_test_vectorized)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')

print(f'Test Accuracy: {test_accuracy:.2f}')
print(f'Test Precision: {test_precision:.2f}')
print(f'Test Recall: {test_recall:.2f}')
print(f'Test F1-score: {test_f1:.2f}')

def home(request):
    return render(request, 'home.html')

def get_answer(request):
    if request.method == 'POST':
        question = request.POST.get('question')

        # Convert the preprocessed question to a feature vector using TF-IDF vectorization
        question_vector = vectorizer.transform([question])

        # Predict the answer using the trained model
        answer = model.predict(question_vector)[0]

        # Return the predicted answer as JSON response
        return JsonResponse({'answer': answer})

