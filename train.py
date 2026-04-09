import pandas as pd
import string
import pickle
import kagglehub
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download dataset
path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")

# Load dataset
file_path = os.path.join(path, "SMSSpamCollection")
df = pd.read_csv(file_path, sep='\t', names=["label", "message"])

# Clean text
def clean_text(text):
    text = text.lower()
    return "".join([c for c in text if c not in string.punctuation])

df['message'] = df['message'].apply(clean_text)

# Convert labels
df['label'] = df['label'].map({'ham':0, 'spam':1})

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Accuracy
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print("Model ready ✅")
