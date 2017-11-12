
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

MAX_SEQUENCE_LENGTH = 100 # 每条新闻最大长度
EMBEDDING_DIM = 200 # 词向量空间维度
VALIDATION_SPLIT = 0.16 # 验证集比例
TEST_SPLIT = 0.2 # 测试集比例 

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_test)
sequences = tokenizer.texts_to_sequences(all_test)

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(all_labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)


p1 = int(len(data)*(1-VALIDATION_SPLIT-TEST_SPLIT))
p2 = int(len(data)*(1-TEST_SPLIT))
x_train = data[:p1]
y_train = labels[:p1]
x_val = data[p1:p2]
y_val = labels[p1:p2]
x_test = data[p2:]
y_test = labels[p2:]
print 'train docs: '+str(len(x_train))
print 'val docs: '+str(len(x_val))
print 'test docs: '+str(len(x_test))

作者：王岳王院长
鏈接：http://www.jianshu.com/p/e21b570a6b8a
來源：簡書
著作權歸作者所有。商業轉載請聯繫作者獲得授權，非商業轉載請註明出處。


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
