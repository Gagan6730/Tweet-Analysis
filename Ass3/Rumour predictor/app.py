# import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import tweepy
# import string
# import nltk
# from nltk.corpus import stopwords
# import pandas as pd
# from text_process import text_process
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline

app = Flask(__name__)


# pickle.dump(vectorMatrix, open('vectorMatrix.pkl', 'wb'))
# pickle.dump(tfidf, open('tfidf.pkl', 'wb'))

# def text_process(mess):
#     no_punc = [c for c in mess if c not in string.punctuation]
#     no_punc = ''.join(no_punc)
#     return [word for word in no_punc.split(' ') if word.lower() not in stopwords.words('english')]

model = pickle.load(open('../model.pkl', 'rb'))
vectorMatrix = pickle.load(open('../vectorMatrix.pkl', 'rb'))
tfidf = pickle.load(open('../tfidf.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''


    apiKey = 'CXCElrPfifVqjfRztp8weameD'
    apiSecretKey = 'o0rZh2qpcXzbiKRoCnZNvYUPmea8miDzb9nNgGQYNNpUlpcC8h'
    auth = tweepy.OAuthHandler(apiKey, apiSecretKey)
    accessToken = '1214429579030319105-Fgrg7ubSr1uXl1IHUz5Q2oYxQhRbAL'
    accessSecretToken = 'dIdmEGnHq6uG9XiIoO1TuNukQM0N8Rz7F5Z2GdBvQbAnk'

    auth.set_access_token(accessToken, accessSecretToken)
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True,
                 retry_count=10, retry_delay=5, retry_errors=set([503]))


    # messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t',
    #                        names=["label", "message"])
    
    # vectorMatrix = CountVectorizer(analyzer=text_process)


    # messageVector = vectorMatrix.fit_transform(messages['message'])

    # #tfidf
    # tfidf = TfidfTransformer()
    # message_tfidf = tfidf.fit_transform(messageVector)

    #train test split
    # msg_train, msg_test, label_train, label_test = train_test_split(
    #     message_tfidf, messages['label'], test_size=0.33)


    # fitting on train data
    # spam_detect_model = MultinomialNB().fit(message_tfidf, messages['label'])

    #app.py from here
    #message from form
    selection = [request.form['select']]
    
    message = [request.form['message']]
    text=[]
    
    val=0
    if selection[0] == 'tweetId':
        try:
            val=int(message[0])
        except:
            return render_template('index.html', prediction_text='Invalid Tweet ID!')
        tweet=""
        try:
            tweet= api.get_status(int(message[0]), tweet_mode="extended")
        except:
            return render_template('index.html', prediction_text='Tweet does not exist!')
        try:
            text.append(tweet.retweeted_status.full_text)
        except:
            text.append(tweet.full_text)
    else:
        text=message

   

    # #creating vector and tdif
    # vectorMatrix = CountVectorizer(analyzer=text_process)
    # tfidf = TfidfTransformer()

    #tra
    #nsforming data
    tweet_text=text[0]
    vector = vectorMatrix.transform(text)
    data = tfidf.transform(vector).toarray()

    #predictiong through the model
    predictions = model.predict(data)
    
    if predictions[0]==0:
        return render_template('index.html', prediction_text='\"{}\"{} is {}'.format(tweet_text,'\n\n',"not a Rumour"))
    else:
        return render_template('index.html', prediction_text='\"{}\"{} is {}'.format(tweet_text,'\n\n',"a Rumour"))




if __name__ == "__main__":
    app.run(debug=True)
