import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import os

# 1. Load cleaned data
df = pd.read_csv('data/cleaned/cleaned_final_names.csv')
df['name'] = df['name'].str.lower()

# 2. Load model and vectorizer
model = joblib.load('model/nationality_model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

# 3. Prepare test set
X = vectorizer.transform(df['name'])
y = df['origin']
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Predict
y_pred = model.predict(X_test)

# 5. Classification Report
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_text = classification_report(y_test, y_pred)

print("📊 Classification Report:\n", report_text)

# Save report
os.makedirs("results", exist_ok=True)
with open("results/report.txt", "w") as f:
    f.write(report_text)

# 6. Confusion Matrix
labels = sorted(df['origin'].unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)

# Save confusion matrix plot
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.close()

# 7. Top performing origins (by recall)
class_performance = {
    label: metrics["recall"]
    for label, metrics in report_dict.items()
    if label in labels
}
top_labels = sorted(class_performance.items(), key=lambda x: x[1], reverse=True)[:10]
top_names, top_recalls = zip(*top_labels)

# Save bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=top_recalls, y=top_names, palette='viridis')
plt.xlabel("Recall")
plt.title("Top 10 Nationalities Predicted Correctly")
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig("results/top_nationalities_recall.png")
plt.close()

print("Saved confusion matrix, report, and top performing nationalities plot in 'results/'")
