from django.shortcuts import render
from django.http import JsonResponse

# Import necessary libraries for machine learning
#کتابخانه های لازم ماشین لرنینگ
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset for training
#لود دیتاست
def read_dataset():
    df = pd.read_csv('dataset.csv')
    return df


df = read_dataset()

# Split the dataset into training and testing sets
# ترین و تست دیتاست
X_train, X_test, y_train, y_test = train_test_split(df['question'], df['answer'], test_size=0.2, random_state=42)

# Create feature vectors using TF-IDF vectorization
#ساخت فیچر
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train the machine learning model
# ترین مدلهای ماشین لرنینگ
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# Evaluate the model on the testing set

y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-score: {f1:.2f}')

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
