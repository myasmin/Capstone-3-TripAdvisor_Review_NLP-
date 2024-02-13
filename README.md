# An NLP Project Report On Analyzing Sentiments of Restaurant Reviews provided by TripAdvisor

The Analysis focuses  on reviews given for restaurants in Barcelona collected from TripAdvisor. It is conducted by adapting NLP Techniques to adapt a meaningful interpretation by analyzing the sentiment of the reviewers on the restaurants.

This dataset is provided by @TripAdvisor, and can be found in [here](https://www.kaggle.com/datasets/inigolopezrioboo/a-tripadvisor-dataset-for-nlp-tasks) in Kaggle. Due to memory allocation, for this analysis the 'Barcelona' dataset was analyzed and runtime resources used in Google Colab platform.

## Context:
Online Reviews have a huge impact on Restaurant and Cafe Businesses. Platforms like TripAdvisor play a crucial role to impact a Customer’s Decision and a Restaurant’s Reputation. The Reviews posted online offer both opportunities and risk, so it is important to analyze the Sentiment of the reviewers. This might contribute to making Restaurateur’s business decisions more precise.

## CRITERIA FOR SUCCESS:
Our goal is to find out the Sentiment behind the reviews on the Restaurants in Barcelona city posted by the reviewers. This is a form implementing Supervised Learning (Sentiment Analysis) based on Keywords (Topics) derived from Unsupervised Learning (Topic Modelling).

## ABOUT THE DATASET:
The source of the data is [here](https://www.kaggle.com/datasets/inigolopezrioboo/a-tripadvisor-dataset-for-nlp-tasks). As mentioned in this Kaggle website- this dataset contains 6 tables for 6 different cities (London, New York, New Delhi, Paris, Barcelona and Madrid), in CSV format. For our analysis, we will consider the ‘Barcelona’ Dataset.
* The ‘Barcelona’ table (426641 rows, 13 columns) contains information on all 426641 customer reviews from from allover in Barcelona
* Each record represents one customer review, and contains details about restaurant name, rating review (1 to 5), sample (positive or negative), full review, date, city etc.

## OBJECTIVE OF THIS STUDY:
Our interest is focused on the Sentiment Analysis of the reviews in the Dataset. The “sample - positive or negative” Column is our event of interest. So, this is a classification problem based on Natural Language Processing. To get a more precise idea, we will apply Topic Modelling to identify the groups (of keywords) and try to find if there is any association between the groups/ topics with the sentiment.

In other words, this problem is an application of Unsupervised Learning (Topic Modelling) to deal the curse of high-dimensionality of a Corpus. Then, we applied Supervised Learning (Classification of Sentiments) to find the intfluentials among the Topics.

## Analysis Overview:
This study has scopes of tuning in terms of grid search CV, hyperparatemer tuning (especially Number of Topic Determination) and also scope for Neural Network application. Overall, I found this analysis a different way to deal with dimensinality reduction, Topic words selection and predictive analysis.

## Tools and Software: 
Python 3.9 (Most Commonly libraries listed below):
Numpy,
Pandas,
Matplotlib,
Seaborn,
NLTK,
Scikitlearn
Genism
Spacy.

