## Summary
See the full implementation here: [movie_rating_predictor.py](movie_rating_predictor.py)

What this script does:

Creates a small sample dataset with movie descriptions + ratings. Converts ratings into High (1) or Low (0) categories. 
Uses TF-IDF to turn text into features. Trains a Logistic Regression model. Prints training & testing accuracy. Predicts a new, unseen movie description.


# Predicting-Movie-Ratings
Predict whether a movie will be rated “high” or “low” based on its description and genre using AI.  Uses a simple machine learning classifier (like Logistic Regression or KNN).
Final project for the Building AI course

Summary
An AI tool that predicts whether a movie will receive a high or low rating based on its description and genre. This helps users quickly decide if a movie might be worth watching.
Building AI course project

Background
Problems:
People spend a lot of time choosing movies they may not enjoy.
Movie descriptions and genres don’t always make clear whether a movie is good.
Streaming platforms often give too many options without helpful filtering.

Why this matters:
Millions of people watch movies daily.
Better recommendations save time and increase satisfaction.

My motivation:
I love movies and often find it hard to choose what to watch.
I wanted to practice using AI for text-based predictions.

How is it used?
Users: Movie enthusiasts, streaming platforms, or casual viewers.
How: User enters a movie description (or selects genre/year).
What happens: The AI predicts if the movie is likely to have a “high” rating (>7) or “low” rating (≤7).
When: Before watching a movie to help decide.

Data sources and AI methods
Data sources:
Public IMDb datasets from Kaggle
Alternatively, a small custom dataset with descriptions and ratings.

AI methods:
Natural Language Processing (TF-IDF for text features).
Classification using Logistic Regression or K-Nearest Neighbors.

Example code snippet:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

data = pd.DataFrame({
    'description': [
        "A thrilling adventure in space",
        "A boring drama about office life",
        "An exciting action-packed movie",
        "A dull and slow romantic movie"
    ],
    'rating': [8, 4, 9, 5]
})

# Convert to high (1) or low (0) rating
data['high_rating'] = (data['rating'] > 7).astype(int)

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(data['description'])
y = data['high_rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

print("Training accuracy:", model.score(X_train, y_train))
print("Testing accuracy:", model.score(X_test, y_test))

Challenges
AI can’t predict personal taste (e.g., someone may love a “low-rated” movie).
Predictions rely heavily on training data quality.
Text preprocessing (removing stop words, handling synonyms) is tricky.
Ethical concern: labeling movies as “low” might discourage people from trying unique films.

What next?
Include more features: actors, directors, year, budget.
Build a simple web app where users paste a movie description.
Extend into a movie recommendation system (suggest similar movies).
Collaborate with designers for a user-friendly interface.

Acknowledgments
IMDb Dataset on Kaggle
Python libraries: scikit-learn, pandas, numpy.
Inspiration: Recommendation systems used by Netflix and IMDb.
