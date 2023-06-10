# Tweets-Sentiment-Analysis

This project focuses on performing sentiment analysis on a dataset of 1.6 million tweets that have been labeled with sentiment (0 for negative, 2 for neutral and 4 for positive) and retrieved using Twitter API.

**Link:** https://www.kaggle.com/datasets/kazanova/sentiment140

**Dataset Name:** training.1600000.processed.noemoticon.csv

## Project Overview

The project is structured as follows:

1. **Dataset**: The Sentiment140 dataset is downloaded from Kaggle, providing a large and diverse set of tweets for sentiment analysis using twitter API.

2. **Exploratory Data Analysis (EDA)**: Exploring the dataset to gain insights into its structure, distribution, and characteristics. This step involves analyzing the distribution of positive and negative sentiments and other relevant statistics.

3. **Data Preprocessing**: Cleaning and preparing the dataset for analysis. This step involves converting text to lowercase, removing stopwords, replacing URLs/Twitter handles/ and performing other preprocessing techniques.

4. **Feature Engineering**: Extracting relevant features from the test using TF-IDF vectorization to represent the text data as numerical features.

5. **Model Evaluation**: Splitting the dataset into training and testing sets, training a Logistic Regression model, and evaluating its performance using metrics such as confusion matrix and classification report.

6. **Prediction**: Using the trained model to predict the sentiment of new, unseen tweets.

7. **Recommended Future Work**: Suggesting potential areas for further improvement and exploration including trying different machine learning, fine-tuning hyperparameters and incorporating advanced techniques such as deep learning.

## Libraries Used

The following libraries were used in this project:

- `re`: Regular expression operations for text preprocessing.
- `numpy`: Numerical computing library for array operations.
- `pandas`: Data manipulation and analysis library for handling datasets.
- `seaborn`: Data visualization library for creating attractive and informative statistical graphics.
- `wordcloud`: Library for generating word clouds from text data.
- `matplotlib`: Data visualization library for creating plots and charts.
- `nltk`: Natural Language Toolkit for text processing and analysis.
- `sklearn`: Machine learning library for various classification tasks.


## Getting Started

To get started with the project, follow these steps:

1. Clone the repository: `git clone https://github.com/Hafsa-Malik/Tweets-Sentiment-Analysis.git` or directly download the `TweetsSentimentAnalysis.ipynb` notebook.
2. Upload the notebook on Google Colab and run.
