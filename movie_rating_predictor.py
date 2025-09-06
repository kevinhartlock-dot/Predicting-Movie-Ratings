import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Example: small dataset
data = pd.DataFrame({
    'description': [
        "A thrilling adventure in space",
        "A boring drama about office life",
        "An exciting action-packed movie",
        "A dull and slow romantic movie"
    ],
    'rating': [8, 4, 9, 5]
})

# Label: high (>7) or low (<=7)
data['high_rating'] = (data['rating'] > 7).astype(int)

# Features
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(data['description'])
y = data['high_rating']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy
print("Training accuracy:", model.score(X_train, y_train))
print("Testing accuracy:", model.score(X_test, y_test))
