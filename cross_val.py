from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot, text_to_word_sequence
from tensorflow.keras.layers import Conv1D, Dropout, Dense, Flatten, LSTM, MaxPooling1D, Bidirectional, Embedding, AveragePooling1D
from tensorflow.keras.optimizers import Adam, RMSprop, Adamax
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from keras.preprocessing.sequence import pad_sequences
import pickle
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np
import gensim
import keras.backend as K
from keras.utils import plot_model
from nltk.tokenize.treebank import TreebankWordDetokenizer
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
import random
###############################
'''
importing unclean tweets, will do some light preprocessing with custom things I need removed from the text and then will use tokenizing and further preprocessing tools from keras. This will need to be done to ensure my data is clean as it works with two languages (at least)
'''
#importing cs labels and tweets- This will be test set

cs_import = open("prepros_dataset_tweet.pkl", "rb")
cs_labels_import = open("labels_text_dataset.pkl", "rb")
cs_tweets = list(pickle.load(cs_import))
cs_labels = list(pickle.load(cs_labels_import))


# combine train tweets lists and train labels list
tweets = cs_tweets
labels = cs_labels

#random seed
seed = 3
np.random.seed(seed)

#dividing data between training, test and development
train_size = len(tweets)


# no of tweets can be param bc political tweets may be more verbose, also using no. of words as extra input to see if political or not
max_no_tokens = 15   # why though?
label_size = 3

indexes = set(np.random.choice(len(tweets), train_size, replace=False))

x_train = np.zeros((train_size, max_no_tokens), dtype=K.floatx())
y_train = np.zeros((train_size, label_size), dtype=np.int32)


#filling numpy arrays with encodings and labels for cs dataset when it is train and test

for i, index in enumerate(indexes):        
    if i < train_size:
        if int(labels[index]) == 1:
            y_train[i,:] = [1.,0.,0.]
        elif int(labels[index]) == 0:
            y_train[i,:]=[0.,1.,0.]
        else:
            y_train[i,:]=[0.,0.,1.]




#embed_model = gensim.models.Word2Vec.load("embedding_3_D50.model")
embed_model = gensim.models.Word2Vec.load("embedding_11_D100.model")
embedding_size = embed_model.wv.vector_size # I think this refers to the size of the word embeddings
embed_vocab_size = len(embed_model.wv.vocab) + 1 


#brute, ineleguant way to fix x_train/x_test with ncoding from word embeddings 
length_dic = len(embed_model.wv.index2word) 


#this is for cs div when cs dataset used as training and test
for i_t, tweet in enumerate(tweets):
    twt=tweet
    for i_word in range(max_no_tokens):
        try:
            word = twt[i_word]
            if i_t < train_size:
                x_train[i_t,i_word] = embed_model.wv.index2word.index(word)
            
        except:
             if i_t < train_size:
                x_train[i_t,i_word] = length_dic
                           


#preparing embeddings
# doing this so I get my +1 missing vocab. Cannot get it with just embed_model.wv.vectors
embedding_matrix = np.zeros((embed_vocab_size,embedding_size))
for i, vector in enumerate(embed_model.wv.vectors):
    embedding_matrix[i] = vector

global_dropout = 0.3
epochs = 25
scores=[]
score_p = open("scores_11_D100.pkl", "wb")
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
for train, test in kfold.split(x_train, y_train[:,0]):
    model = Sequential()
    model.add(Embedding(embed_vocab_size,embedding_size,weights=[embedding_matrix],input_length=max_no_tokens,trainable=True))
    model.add(Bidirectional(LSTM(128, activation='relu', dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True)))
    model.add(Bidirectional(LSTM(128, activation='relu', return_sequences=False)))
    model.add(Dropout( global_dropout ))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(global_dropout))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout( global_dropout ))
    model.add(Dense(label_size, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adamax(lr=0.0002), metrics=['accuracy'])
    history = model.fit(x_train[train,:],y_train[train,:],batch_size=8, shuffle= False, epochs= epochs ,verbose=1)
    results=model.evaluate(x=x_train[test,:], y=y_train[test,:], batch_size=8, verbose=1)
    scores.append(results[1])
    pickle.dump(scores,score_p)
# cross validate model using 10 step cross validation


print(np.mean(scores),np.std(scores))

#plot model
embedding = "CS Train and Test: CBOW"
# Plot training & validation accuracy values
plt.ion()
plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
plt.title('Model accuracy ' + " " + embedding)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#plot model                                                                                                                                                                                                        
embedding = "cs Test and Train"
# Plot training & validation accuracy values                                                                                                                                                                       
plt.ion()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss ' + " " + embedding)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()


#plot_model(model, to_file='model_7.6.png')
model.save('SA_model_2.model')
