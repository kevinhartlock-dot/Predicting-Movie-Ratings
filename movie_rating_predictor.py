# movie_rating_predictor.py
# Building AI course project - Movie Rating Predictor

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def main():
    # Example dataset (you can replace with a bigger one later)
    data = pd.DataFrame({
        'description': [
            "A thrilling adventure in space",
            "A boring drama about office life",
            "An exciting action-packed movie",
            "A dull and slow romantic movie",
            "A fun comedy with clever jokes",
            "A depressing and overly long war film"
        ],
        'rating': [8, 4, 9, 5, 7, 3]
    })

    # Convert numeric rating into binary: high (1) if > 7, low (0) if <= 7
    data['high_rating'] = (data['rating'] > 7).astype(int)

    # Transform text into TF-IDF features
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(data['description'])
    y = data['high_rating']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Print accuracy
    print("Training accuracy:", model.score(X_train, y_train))
    print("Testing accuracy:", model.score(X_test, y_test))

    # Try predicting a new description
    new_movie = ["A suspenseful mystery with shocking twists"]
    new_features = tfidf.transform(new_movie)
    prediction = model.predict(new_features)
    print("\nNew movie description:", new_movie[0])
    print("Predicted rating category:", "High" if prediction[0] == 1 else "Low")

if __name__ == "__main__":
    main()
