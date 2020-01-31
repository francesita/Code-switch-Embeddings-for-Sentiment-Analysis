import tweepy
import pickle
import sys

'''
Method for extracting code-switched in Spanish and English from twitter.
The key word fle contains words in Spanish
'''

consumer_token = open( "private/.consumer_token" ).read().strip()
consumer_secret = open("private/.consumer_secret").read().strip()

access_token = open("private/.access_token").read().strip()
access_token_secret = open("private/.access_secret").read().strip()

auth = tweepy.OAuthHandler(consumer_token, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# this constructs API's instance                                                                                                                                                                                   
api = tweepy.API(auth, wait_on_rate_limit = True)

#importing the dictionary that exists and that will be updated if new tweets are added
# If starting from scratch, uncomment dic={} and comment out loading of pickle_file
pickle_file = open("cs_tweets.pkl","rb")
dic = pickle.load(pickle_file)
#dic={}

updated_dic_file = open("cs_tweets.pkl", "wb")
#backup = open("data/cs_dic_update_10.1.pkl", "wb")
#variables corresponding to files so that nothing is overwritten
def find_tweet(input_file):
    read_file = open(input_file)
    for line in read_file:
        q = line
        extract_tweet(q)
        pickle.dump(dic, updated_dic_file)

def extract_tweet(query):
    print(query)
    try:
        for tweet in tweepy.Cursor(api.search, q=query, count=100, lang="en",tweet_mode = "extended", status="2019-08-26").items(5000):
        #checking if tweet is a retweet and if value exists in dictionaru to avoid duplicates
            if hasattr(tweet, 'retweeted_status') and tweet.retweeted_status.id in dic:
                continue
        #checking if tweet is a retweet and if value exists in dictionaru to avoid duplicates
            elif  tweet.id in dic:
                continue
            elif hasattr(tweet, 'retweeted_status'):
                dic[tweet.retweeted_status.id] = tweet.retweeted_status.full_text
                cs_file.write(tweet.retweeted_status.full_text)
            else:
                dic[tweet.id] = tweet.full_text
                cs_file.write(tweet.full_text + "\n")
    except tweepy.TweepError as e:
        print(e.reason)
        print(query)
        find_tweet("key_words.txt")
        
find_tweet("key_words.txt")



