# An NLP Project Report On Analyzing Sentiments of Restaurant Reviews provided by TripAdvisor

The Analysis focuses  on reviews given for restaurants in Barcelona collected from TripAdvisor. It is conducted by adapting NLP Techniques to adapt a meaningful interpretation by analyzing the sentiment of the reviewers on the restaurants.

This large dataset is provided by @TripAdvisor, and can be found in [here](https://www.kaggle.com/datasets/inigolopezrioboo/a-tripadvisor-dataset-for-nlp-tasks) in Kaggle. Due to memory allocation, for this analysis the 'Barcelona' dataset was analyzed and runtime resources used in Google Colab platform.

## Context:
Online Reviews have a huge impact on Restaurant and Cafe Businesses. Platforms like TripAdvisor play a crucial role to impact a Customer’s Decision and a Restaurant’s Reputation. The Reviews posted online offer both opportunities and risk, so it is important to analyze the Sentiment of the reviewers. This might contribute to making Restaurateur’s business decisions more precise.

## CRITERIA FOR SUCCESS:
Our goal is to find out the Sentiment behind the reviews on the Restaurants in Barcelona city posted by the reviewers. This is a form implementing Supervised Learning (Sentiment Analysis) based on Keywords (Topics) derived from Unsupervised Learning (Topic Modelling).

## ABOUT THE DATASET:
The source of the data is [here](https://www.kaggle.com/datasets/inigolopezrioboo/a-tripadvisor-dataset-for-nlp-tasks). As mentioned in this Kaggle website- this dataset contains 6 tables for 6 different cities (London, New York, New Delhi, Paris, Barcelona and Madrid), in CSV format. For our analysis, we will consider the ‘Barcelona’ Dataset.
* The ‘Barcelona’ table (426641 rows, 13 columns) contains information on all 426641 customer reviews from from allover in Barcelona
* Each record represents one customer review, and contains details about restaurant name, rating review (1 to 5), sample (positive or negative), full review, date, city etc.

## OBJECTIVE OF THIS STUDY:
Our interest is focused on the Sentiment Analysis of the reviews in the Dataset. The “sample - positive or negative” Column is our event of interest. So, this is a classification problem based on Natural Language Processing analysis. Our analysis was to apply Topic Modelling to identify the groups (of keywords) and try to find if there is any association between the groups/ topics with the sentiment.

In other words, this problem is an application of Unsupervised Learning (Topic Modelling) to deal the of high-dimensionality. Then, we applied Supervised Learning (Classification of Sentiments) to find the intfluentials among the Topics.

## ANALYSIS OVERVIEW:
This analysis is an implementation of Unsupervised Modelling technique (Topic Modelling) on Supervised Modelling (Classification of Sentiments). Rather than PCA or any other Technique, Topic Modelling is implemented to reduce the dimensionality of the Corpus. I found this approach more appealing and stronger than PCA. After reducing the dimentionality, the identified topics and other factors are used to Classify the Sentiments.

![Positive Wordcloud](https://github.com/myasmin/Capstone-3-TripAdvisor_Review_NLP-/blob/main/plots/positive%20wordcloud.png)

![Negative Wordcloud](https://github.com/myasmin/Capstone-3-TripAdvisor_Review_NLP-/blob/main/plots/negative%20wordcloud.png)

![](https://github.com/myasmin/Capstone-3-TripAdvisor_Review_NLP-/blob/main/plots/frequency%20of%20top%20words.png)

The Stopwords selection criteria was crucial. Looking into the frequency of the words, selecting the ricght words were important. The words like *'good', 'great', 'service'* etc played important role in model performances (in terms of positive reviews especially). In the otherhand, selecting the other high frequency words were important, but not impactful on sentiment classification (*"food","place","restaurant"*).

6 Topics were selected in the Corpus, and 2 of them were found to be most dominant.

![](https://github.com/myasmin/Capstone-3-TripAdvisor_Review_NLP-/blob/main/plots/wordcloud%20by%20topics.png)

![](https://github.com/myasmin/Capstone-3-TripAdvisor_Review_NLP-/blob/main/plots/WC%20and%20important%20topic%20keywords.png)

Our Analysis showed that **Topic 5** and **Topic 3** are the most dominant topics for the Barcelona Dataset.

![](https://github.com/myasmin/Capstone-3-TripAdvisor_Review_NLP-/blob/main/plots/number%20of%20topics%20by%20topic%20keywords.png)


**Topic 5: good, great, recommend, friendly, nice, really, well, tapa, wine, menu, excellent, also, dish, atmosphere, try, delicious, definitely, love, small, taste, little, quality, tasty, eat, fresh, highly, lovely, bit, choice, different, lot, selection, dessert, cook, tapas, choose, meat, salad, quite, special**


**Topic 3: get, make, come, order, back, even, eat, waiter, drink, take, want, try, say, friend, see, feel, give, think, pizza, ever, people, leave, last, always, way, ask, know, full, family, absolutely, thing, start, end, owner, decide, away, speak, still, pay, happy**

Then came the part of Sentiment Analysis. We had some interesting findings:

![](https://github.com/myasmin/Capstone-3-TripAdvisor_Review_NLP-/blob/main/plots/sample%20rating%20review%20distribution.png)


![Wordcount in Reviews](https://github.com/myasmin/Capstone-3-TripAdvisor_Review_NLP-/blob/main/plots/rating%20review%20by%20wordcount.png)

We observed that, even reviewers are likely to leave positive reviews, but people use more word for negative reviews in their comments.

In the stage of Sentiment Analysis, we found that the wordcount, Topic 5 and Topic 3 are the most influential factors while classifying sentiments.

![](https://github.com/myasmin/Capstone-3-TripAdvisor_Review_NLP-/blob/main/plots/feature%20importance%20by%20factors.png)

The Random Forest and Logistic Regression models performed well in terms of accuracy (RF 58.25% accurate and Logistic 59.05%).But, we would also emphasize on the F1 score since our count of false positives and false negatives.

This study has scopes of tuning in terms of grid search CV, hyperparatemer tuning (especially Number of Topic Determination) and also scope for Neural Network application for LLM. 

Overall, I found this analysis a different way to deal with dimensinality reduction, Topic words selection and predictive analysis, which has more stronger, faster and scientific approach for Sentiment Analysis in NLP projects.

Details on Data Wrangling, EDA and Preprocessing [here](https://github.com/myasmin/Capstone-3-TripAdvisor_Review_NLP-/blob/main/Data_Preprocessing_and_EDA.ipynb).

Details on Analysis and Modelling [here](https://github.com/myasmin/Capstone-3-TripAdvisor_Review_NLP-/blob/main/Analysis_and_Modelling.ipynb).

Find a project report in [here](https://github.com/myasmin/Capstone-3-TripAdvisor_Review_NLP-/blob/main/Project%20Report.pdf).


## Tools and Software: 

Python 3.9 (Most Commonly libraries listed below):
* Numpy,
* Pandas,
* Matplotlib,
* Seaborn,
* NLTK,
* Scikitlearn
* Genism
* Spacy.

## Acknowlegdement:

* https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/
* https://eugenia-anello.medium.com/nlp-tutorial-series-d0baaf7616e0
* https://medium.com/nanonets/topic-modeling-with-lsa-psla-lda-and-lda2vec-555ff65b0b05
* https://www.kaggle.com/datasets/inigolopezrioboo/a-tripadvisor-dataset-for-nlp-tasks
* https://zenodo.org/records/6583422

Data:
* https://zenodo.org/records/6583422/files/Barcelona_reviews.csv?download=1
  


