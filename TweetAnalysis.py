#!/usr/bin/env python
import requests
import re
import nltk
import twitter
import csv
import math
import sklearn.ensemble
import sklearn.metrics
import sklearn
import numpy as np
import pickle
import os
import random
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

kTestTrainingSplitRatio = 0.3

class TweetAnalysis:
  def __init__(self):
    self.twitter = twitter.Twitter(auth=twitter.OAuth('2223635090-IaNrBG3WiiPnWTVtxupFCgB5TXAtlzKeQYRdrKu',
                                 'PDfV4DDZjB4T2EbnVAzsZCdDUMNJYDPLYlYErfuhcTNPE',
                                 'nrZxSV89t9pvZsbWHjXs7Y8vZ',
                                 'FQA4pFzrPWgVCbsUu5nXi5eWbH2T1gD7bkDAogtRmyTjzUjxNO'))
    # container for raw training data, loaded from csv
    self.training_data = []
    self.test_data = []

    # api response object (dict extracted from api json response)
    self.api_response_raw = None
    # grab list of stopwords from nltk, returns list of strings
    self.stopwords = nltk.corpus.stopwords.words('english')
    # add some extra stop words specific to our use case
    self.addExtraStopWords()
    
  def addExtraStopWords(self):
    extra_stopwords = ["and"]
    for sw in extra_stopwords:
      self.stopwords.append(sw.lower())

  # Set raw data from outside class
  def addRaw(self, api_response):
    self.api_response_raw = api_response
    self.extractRaw()

  # Convert raw response into a data structure that we can use more easily
  # Here we choose a list of dictionaries where each dictionary contains relevant info about each tweet
  def extractRaw(self):
    # iterate over each raw tweet in the api response and save just the data we want
    cnt = 0
    tweets = []
    for r in self.api_response_raw['statuses']:
      tweets.append(
        {
          "user":str(r['user']['screen_name'].encode("utf-8")),
          "text":str(r['text'].encode("utf-8")),
          "name":str(r['user']['name'].encode("utf-8")),
          "retweets":r['retweet_count'],
          "favorites":r['favorite_count'],
          "created_at":r['created_at'],
          "followers_count":r['user']['followers_count'],
          "sentiment":None
        })
      cnt = cnt + 1
    print("Extracted {} tweets from raw response".format(cnt))
    return tweets

  def fetchRaw(self, query, count):
    print("Fetching data from twitter api with query: {} | count: {}".format(query, count))
    # response is a dictionary that the twitter api returned
    # tweet information is stored within list under the key "statuses" inside the outer most dict
    # see https://dev.twitter.com/rest/reference/get/search/tweets for schema
    self.api_response_raw = self.twitter.search.tweets(q=query, count=count)
    print("Fetched {} raw tweets from api".format(len(self.api_response_raw.get('statuses'))))
    # pass out our tweets list
    return self.extractRaw()

  def clean(self, list_of_tweets):
    if len(list_of_tweets) == 0:
      print("Cannot clean data, tweets list is empty")
      return
    # let's first drop all RTs since we just care about user mentions
    original_cnt = len(list_of_tweets)
    list_of_tweets = [w for w in list_of_tweets if "RT @" not in w['text']]
    print("Removed {} retweets from our tweets list".format(original_cnt - len(list_of_tweets)))

    #print(list_of_tweets)
    self.rawtweets = list(list_of_tweets)
    progress_cnt = 0
    for tweet_dict in list_of_tweets:
      # print("Before clean:\n{}".format(tweet_dict['text']))
      # remove urls from tweet string
      tweet_dict['text'] = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', tweet_dict['text'])
      # strip punctuation
      tweet_dict['text'] = re.sub(r'[^a-zA-Z]', ' ', tweet_dict['text'])
      # strip emojiis
      emoji_pattern = re.compile(
        u"(\ud83d[\ude00-\ude4f])|"  # emoticons
        u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
        u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
        u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
        u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
        "+", flags=re.UNICODE)
      tweet_dict['text'] = emoji_pattern.sub(r'', tweet_dict['text'])
      # make all words lowercase
      tweet_dict['text'] = tweet_dict['text'].lower()
      tweet_text_split = tweet_dict['text'].split()
      # remove stop words from nltk corpus
      # tweet_text_split = [w for w in tweet_text_split if not w in self.stopwords]
      # remove user if in the tweet string
      tweet_dict['text'] = " ".join(tweet_text_split)
      # print(tweet_text_split)
      # print("After clean:\n{}".format(tweet_dict['text']))
      if progress_cnt % 10000 == 0 and progress_cnt > 1: 
        print("Cleaning tweet {} of {} | {:0.1f} %".format(progress_cnt, len(list_of_tweets), (float(progress_cnt)/float(len(list_of_tweets))*100)))
      progress_cnt = progress_cnt + 1
    print("Cleaned {} tweets".format(len(list_of_tweets)))
    return list_of_tweets

  def loadTrainingData(self, filepath, maxNum=-1):
    pickle_filepath = filepath + ".pickle"
    if not os.path.isfile(pickle_filepath):
      print("Loading training data from {}, please wait...".format(filepath))
      with open(filepath) as f:
        for i,row in enumerate(csv.DictReader(f, skipinitialspace=True)):
          if maxNum > 0 and i == maxNum:
            break
          self.training_data.append(row)
      print("Loaded training data: {} tweets".format(len(self.training_data)))
      # clean our training data
      print("Cleaning training data set. This could take a while...")
      self.training_data = self.clean(self.training_data)
      # dump the training data to a pickle file so we dont have to do this every time
      with open(pickle_filepath, "wb") as f:
        pickle.dump(self.training_data, f, pickle.HIGHEST_PROTOCOL)
      print("Wrote training data to pickle: {}".format(pickle_filepath))
    else:
      print("Found pickle file with training data.")
      print("Loading training set from: {}".format(pickle_filepath))
      self.training_data = pickle.load(open(pickle_filepath, "rb"))

    # split the training data into test and training sets
    random.shuffle(self.training_data)
    size_of_dataset = len(self.training_data)
    index_for_split = int(size_of_dataset * kTestTrainingSplitRatio)
    print("Splitting dataset into test/training by ratio {} at index {} of {}".format(kTestTrainingSplitRatio, index_for_split, len(self.training_data)))
    self.test_data = self.training_data[:index_for_split]
    self.training_data = self.training_data[index_for_split:]

  # Train classifier
  def buildTrainingModel(self):
    train_data = []
    train_labels = []
    for d in self.training_data:
      train_data.append(d['text'])
      train_labels.append(d['sentiment'])
    print("Training classifier with {} items".format(len(train_data)))
    self.vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(min_df=1, max_df = 0.8, sublinear_tf=True, use_idf=True, decode_error='ignore')
    train_vectors = self.vectorizer.fit_transform(train_data)
    ## Random Forest has 69% accuracy
    # self.classifier = RandomForestClassifier()
    ## MultinomialNB has 73% precision rate
    # self.classifier = MultinomialNB()
    ## SGD 76% accuracy
    self.classifier = SGDClassifier()
    self.classifier.fit(train_vectors, train_labels)
    print("Done training classifier!")
    # save pickle files
    vectorizer_path = "vectorizer.pickle"
    classifier_path = "classifier.pickle"
    with open(vectorizer_path, "wb") as f:
        pickle.dump(self.vectorizer, f, pickle.HIGHEST_PROTOCOL)
    print("Wrote vectorizer to pickle: {}".format(vectorizer_path))
    with open(classifier_path, "wb") as f:
        pickle.dump(self.classifier, f, pickle.HIGHEST_PROTOCOL)
    print("Wrote classifier to pickle: {}".format(classifier_path)) 

  def testTrainingModel(self):
    print("Testing {} known data points against classifier".format(len(self.test_data)))
    test_data = []
    test_labels = []
    for d in self.test_data:
      test_data.append(d['text'])
      test_labels.append(d['sentiment'])
    test_vectors = self.vectorizer.transform(test_data)
    test_prediction = self.classifier.predict(test_vectors)
    print(sklearn.metrics.classification_report(test_labels, test_prediction))

  def run(self, list_of_dicts):
    if type(list_of_dicts) is dict:
      list_of_dicts = [list_of_dicts]
    list_of_text = []
    for t in list_of_dicts:
      list_of_text.append(t["text"])
    x_test = self.vectorizer.transform(list_of_text)
    results = self.classifier.predict(x_test)
    print("=====================")
    for i,t in enumerate(list_of_text):
      print("{} -> {}".format(list_of_text[i], results[i]))
    print("=====================")

  def loadClassifier(self, vectorizer_path = "vectorizer.pickle", classifier_path = "classifier.pickle"):
    if os.path.isfile(vectorizer_path) and os.path.isfile(classifier_path):
      print("Previous classifier found on disk!")
      print("Loading vectorizer from: {}".format(vectorizer_path))
      self.vectorizer = pickle.load(open(vectorizer_path, "rb"))
      print("Loading classifier from: {}".format(classifier_path))
      self.classifier = pickle.load(open(classifier_path, "rb"))
      return True
    else:
      return False

if __name__ == "__main__":
  ta = TweetAnalysis()
  # attempt to load classifier from previous run
  if not ta.loadClassifier():
    ta.loadTrainingData("sentiment_training_set.csv")
    ta.buildTrainingModel()
    ta.testTrainingModel()
  # run on random data
  ta.run({"text":"I love my job!"})
  ta.run({"text":"I hate my job!"})
  # run on twitter stream
  tweets = ta.fetchRaw("@united", 5)
  ta.run(ta.clean(tweets))
