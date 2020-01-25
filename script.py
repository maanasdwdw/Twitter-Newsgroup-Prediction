import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets import fetch_20newsgroups
import tweepy

#creating instance of the class
app=Flask(__name__)
categories = ["talk.politics.misc","misc.forsale","rec.motorcycles",
"comp.sys.mac.hardware","sci.med","talk.religion.misc","talk.politics.guns","talk.politics.mideast","soc.religion.christian","alt.atheism","sci.space","sci.electronics","sci.crypt","rec.sport.baseball",
"rec.sport.hockey","rec.autos","comp.windows.x","comp.graphics",
"comp.os.ms-windows.misc",
"comp.sys.ibm.pc.hardware"]

def bag_of_words(categories):
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(fetch_train_dataset(categories).data)
    pickle.dump(count_vect.vocabulary_, open("./model_vocab/vocab.pickle", 'wb'))
    return X_train_counts

def fetch_train_dataset(categories):
    twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
    return twenty_train

def tf_idf(categories):
    tf_transformer = TfidfTransformer()
    return (tf_transformer,tf_transformer.fit_transform(bag_of_words(categories)))

consumer_key='XXXX'
consumer_secret='XXXX'
access_token='XXXX'
access_token_secret='XXXX'

# Creating the authentication object
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# Setting your access token and secret
auth.set_access_token(access_token, access_token_secret)
# Creating the API object while passing in auth information
api = tweepy.API(auth)

# Creating the API object while passing in auth information
api = tweepy.API(auth)




#to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/index')

def index():
    return flask.render_template('index.html')
def ValuePredictor(to_predict_list):
    to_predict = (to_predict_list)
    vocabulary_to_load = pickle.load(open("./model_vocab/vocab.pickle", 'rb'))
    count_vect = CountVectorizer(vocabulary=vocabulary_to_load)
    load_model = pickle.load(open("./model_vocab/model.pickle", 'rb'))
    count_vect._validate_vocabulary()
    tfidf_transformer = tf_idf(categories)[0]
    X_new_counts = count_vect.transform([to_predict])
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    predicted = load_model.predict(X_new_tfidf)
    return predicted[0]
@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        # The Twitter user who we want to get tweets from
        name = to_predict_list[0]
        # Number of tweets to pull
        tweetCount = 1

        # Calling the user_timeline function with our parameters
        results = api.user_timeline(id=name, count=tweetCount)

        # foreach through all tweets pulled
        for tweet in results:
        # printing the text stored inside the tweet object
            tweet=tweet.text

        result = ValuePredictor(tweet)
        prediction=fetch_train_dataset(categories).target_names[result]
        return render_template("index.html",prediction=prediction)

