# Spam SMS Detector

A machine learning project that classifies SMS messages as spam or ham (not spam).

## Tech Stack
- Python
- pandas
- scikit-learn
- TF-IDF Vectorizer
- Multinomial Naive Bayes

## Dataset
UCI SMS Spam Collection (5,574 messages), from Kaggle.

## Result
Accuracy: 96.68% on the test set.

## How it works
1. Load and split the data (80% train, 20% test)
2. Convert text to numbers using TF-IDF
3. Train a Naive Bayes model
4. Predict whether a new message is spam or ham

## Run it
Open `spam_detector.ipynb` in Google Colab, upload `spam.csv` and run the cell.
