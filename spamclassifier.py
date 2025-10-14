
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import string
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder

nltk.download('stopwords')


data = pd.read_csv(r"C:\Users\hp\Downloads\archive (1)\spam.csv", encoding='latin-1')

#feature selection
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

print("Dataset Loaded Successfully!")
print(data.head())

# Data Cleaning

le = LabelEncoder()
data['label'] = le.fit_transform(data['label'])  # ham => 0, spam => 1

def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    return text

data['message'] = data['message'].apply(clean_text)


X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

#train model

model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Evaluate Model

y_pred = model.predict(X_test_vec)

print(" Model Evaluation:")
print("Accuracy:", round(accuracy_score(y_test, y_pred)*100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#Test on New Email

sample = ["Congratulations! You won a free iPhone. Click the link to claim your prize."]
sample_vec = vectorizer.transform(sample)
prediction = model.predict(sample_vec)

print("Sample Message:", sample[0])
print("Prediction:", "Spam" if prediction[0] == 1 else "Not Spam")

