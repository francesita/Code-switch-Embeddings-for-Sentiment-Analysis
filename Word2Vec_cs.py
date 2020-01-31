import gensim
import pickle
import preprocess
import logging

p = open("data_CS/last_embed.pkl", "rb")
cs_dic = pickle.load(p)


def fix_dictionary(cs_dic):
   # new_dic = cs_dic = {}
    for key in cs_dic:
        v=cs_dic.get(key)
        cs_dic[key] = {}
        cs_dic[key]['text'] = v
    return cs_dic

preprocessed_filename = "preprocessed_data_improved_tweets.pkl"
#this tokenized and preprocesses the data. It saves the data into a pickle file title preprocessed_data
preprocess.preprocess_tweets(fix_dictionary(cs_dic), preprocessed_filename)
##################################################

#begin word2vec stuff with new_dictionary
#opens list created from the above function
f = open(preprocessed_filename, "rb")
clean_tweets = pickle.load(f)
'''
def pop_textfile(new_dic):
    for key in new_dic:
        text_file.write(new_dic.get(key,{}).get('tokens'))
'''
#read_file = open("clean_cs_data.txt", "rb")]

logging.basicConfig(format='%(asctimes)s : %(levelname)s : %(message)s', level=logging.INFO)

# loop tp get all the tokens for each key in the dictionary and add them to the list 'sentences'since that is what needs to be given as the first argument for the word2vec model
'''
for key in new_dic:
    sentences.append(new_dic.get(key,{}).get('tokens'))
'''
documents = clean_tweets
logging.info("finished processing")

model = gensim.models.Word2Vec(documents, size = 100, window=5, min_count=2, workers=10, iter=20, sg=0)
model.save("embedding_11_D50.model")
model.wv.save_word2vec_format('embedding_11_D50.bin', binary = True)
model.train(documents, total_examples=len(documents),epochs=20)


'''
for i, line in enumerate(read_file):
    yield(gensim.utils.simple_process(line))
    
'''
#doo logging stuff to see where you are in the code

'''
for key in cs_dic:
    preprocess.tokenize(cs_dic.get(key))
'''

