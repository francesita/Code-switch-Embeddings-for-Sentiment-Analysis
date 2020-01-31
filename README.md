# Code-switch-Embeddings-for-Sentiment-Analysis

This repository contains:

1. A folder called models two models: es_en model and cs_model
2. A folder called script_train_models two scripts for training models: es_en.py and cs_model.py
    Both of these use code that has been altered from https://github.com/jatinmandav/Neural-Networks Sentiment Analysis section for word2vec.
3. A folder called preprocessing with a script for preprocessing the datasets and another for preprocessing tweet data collected from Twitter and held in a dictionary.
Some of the preprocessing methods here use inspiration from https://github.com/jatinmandav/Neural-Networks Sentiment Analysis section for word2vec.
4. The script used to do 10-step cross validation on the code-switched model called cross_val.py
5. Key word file used for tweet extraction called extract_cs_tweets.py. Tweepy was used for this and some conventions were followed from Tweepy documentations. 
6. Script tweet_ext.py used to extract tweets listed in datasets for SemEval and CS tweets
6. Two TSV files that can be uploaded to tensorflow website for embedding visualization
7. Folder containing preprocessed pickle dictionaries containing datasets to train the models
8. Word2vec file used to make the word embeddings (word2vec_cs.py). A tutorial was followed from https://kavita-ganesan.com/gensim-word2vec-tutorial-starter-code/ and some of the code was used from there.
    Some code was also used and inspired from from https://github.com/jatinmandav/Neural-Networks

