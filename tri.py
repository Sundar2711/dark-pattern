import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Load the dataset
dataset = pd.read_csv('dataset.csv')

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset['text'], dataset['Pattern Category'], test_size=0.2, random_state=42)

# Step 3: Text vectorization using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Step 4: Train the Naive Bayes model
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(X_train_tfidf, y_train)

# Save the model
model_filename = 'dark_pattern_detection_model_naive_bayes.pkl'
joblib.dump(naive_bayes_model, model_filename)

# Now you can use the trained model for predictions
text = "I don't feel lucky"
text_tfidf = tfidf_vectorizer.transform([text])
prediction = naive_bayes_model.predict(text_tfidf)

print(f"Predicted Category: {prediction[0]}")