
testdataPath = '/Users/billxu/Downloads/詩籤與評分 blog - test_data.csv'  
traindataPath = '/Users/billxu/Downloads/詩籤與評分 blog - train_data.csv'  
all_dataPath = '/Users/billxu/Downloads/詩籤與評分 blog - data.csv'

# -*- coding: utf-8 -*-
###################  
# Step2: Download IMDB  
####################  


import keras 
import jieba
import re  
import os
import urllib.request
import logging
import tarfile 
import numpy as np
from keras.preprocessing import sequence  
from keras.preprocessing.text import Tokenizer 
from keras.datasets import reuters
from keras.layers import Dense, Dropout, Activation
  
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'  
logging.basicConfig(format=LOG_FORMAT)  
logger = logging.getLogger('LOG')  

logger.setLevel(logging.DEBUG) 

def rm_tags(text):  
    r'''  
    Remove HTML markers  
    '''  
    re_tag = re.compile(r'<[^>]+>')  
    return re_tag.sub('', text)  
  
def read_files(_dataPath):  
    r'''  
    Read data from IMDb folders  
  
    @param filetype(str):  
        "train" or "test"  
  
    @return:  
        Tuple(List of labels, List of articles)  
    '''  
    file_list = []  
    all_labels = []
    all_texts = []  
    import csv
    f = open(_dataPath, 'r')
    for row in csv.DictReader(f):
        words = jieba.cut_for_search(row['document']    )
        
        all_texts += [" ".join(words)]
        #all_texts += [" "+ row['document']]
        all_labels += row['class']
        #print(all_texts+all_labels) 
    return all_labels, all_texts  
    f.close()

train_labels, train_text = read_files(traindataPath)  
test_labels, test_text = read_files(testdataPath) 
all_labels,all_test = read_files(all_dataPath) 

###################  
# Step3: Tokenize  
####################  

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_test)
sequences = tokenizer.texts_to_sequences(train_text)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = tokenizer.sequences_to_matrix(sequences, mode='tfidf')

labels = keras.utils.to_categorical(np.asarray(train_labels))
labels_test = keras.utils.to_categorical(np.asarray(test_labels))


max_words = 1000
batch_size = 512
epochs = 10

###################  
# Step4: Building MODEL  
####################  
from keras.models import Sequential  
from keras.layers.core import Dense, Dropout, Activation, Flatten  
from keras.layers.embeddings import Embedding  
from keras.models import model_from_json
from keras.layers.recurrent import SimpleRNN, LSTM

model = Sequential()

model.add(Dense(512, input_shape=(len(word_index)+1,), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(labels.shape[1], activation='softmax'))
model.summary()

###################  
# Step5: Training  
###################  

logger.info('Start training process...')  

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(data, labels,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_split=0.5)


'''
###################  
# Step6: Evaluation  
###################  
'''
test_sequences = tokenizer.texts_to_sequences(test_text)
test_data = tokenizer.sequences_to_matrix(test_sequences, mode='tfidf')

logger.info('Start evaluation...')  
scores = model.evaluate(test_data, labels_test, verbose=1)  
print("")  
logger.info('Score={}'.format(scores[1]))  
