import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib
import os

# 1. Load the cleaned dataset
df = pd.read_csv('data/cleaned/cleaned_final_names.csv')

# 2. Clean and preprocess
df.dropna(subset=['name', 'origin'], inplace=True)
df['name'] = df['name'].str.lower()

# 3. Vectorize names using character n-grams
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
X = vectorizer.fit_transform(df['name'])
y = df['origin']

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# 7. Save model and vectorizer
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/nationality_model.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')

print("Model and vectorizer saved to 'model/' directory.")
