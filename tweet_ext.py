import tweepy
import nltk
import sys
import pickle
# Practice to have tweets from tweetID
"""
This code reads from a textfile containing tweetId's into tweets.
It extracts the tweets from twitter and writes them onto a new textfile

@author: Frances Adriana Laureano De Leon
@date: 2019-06-25
"""

consumer_token = open( "private/.consumer_token" ).read().strip()
consumer_secret = open("private/.consumer_secret").read().strip() 

access_token = open("private/.access_token").read().strip()
access_token_secret = open("private/.access_secret").read().strip()

auth = tweepy.OAuthHandler(consumer_token, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# this constructs API's instance 
api = tweepy.API(auth, wait_on_rate_limit = True)


readf = open("datasets/SemEval2014-Task9-subtaskAB-test-to-download/SemEval2014-task9-test-B-gold-NEED-TWEET-DOWNLOAD.txt", "r")
dictionary = {}
#writef = open("full_tweets.txt", "w")

for line in readf:
    # line = [ int(s) for s in line.split() if s.isdigit()]    #extracting the digit only since dataset contains
    print( line )
    #extra_Stuff only used for semeval test set text file bc one extra thing to unpack
    try:
        tweet_id, ignore, sentiment, extra_stuff = line.split( "\t")
    except:
        tweet_id, ignore, sentiment = line.split( "\t")
    # change needed due to way dataset is given semeval
    try:
        tweet_id = int( tweet_id )
    except:
        tweet_id = -1
    dictionary[tweet_id] = {}
    dictionary[tweet_id]['sentiment'] = sentiment
    dictionary[tweet_id]["Fail"]      = False
    
    #finds tweet with tweet_id and writes it onto writef
    try:
        tweet = api.get_status(tweet_id, tweet_mode = "extended")
        if hasattr(tweet, "retweeted_status"):
            t = tweet.retweeted_status.full_text
            dictionary[tweet_id]['text'] = t
        else:
            t = tweet.full_text  # this gets the text portion of the tweet
            # t_encode = t.encode('utf-8')
            dictionary[tweet_id]['text'] = t
            #adding tweet tokenized as token in dictionary
        dictionary['token'] = nltk.word_tokenize(t)
        #writef.write(t.encode('ascii', 'backslashreplace') + '\n') # ascii characters cause an error so maybe there is something that can be improved upon there, I may need emojis to analyze sentiment
    except tweepy.TweepError as e:
          # if dictionary[tweet_id]["Fail"] = True:
        #print(e.args[0][0]['message'])
        #print e.response
        #print e.reason
        dictionary[tweet_id]["Fail"] = True
        dictionary[tweet_id]["response"] = e.response

write_file = open("semeval_dataset_test_dic.pkl", "wb")
pickle.dump(dictionary, write_file)
write_file.close()
"""
remove \n from tweets before saving

"""
