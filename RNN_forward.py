import nltk
from nltk.corpus import gutenberg
from nltk.corpus import brown
from nltk.corpus import reuters

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM, Input, Embedding, merge
from keras.optimizers import Adamax
from keras.regularizers import l2

from collections import defaultdict
from collections import Counter

import numpy as np
import pickle


winSize = 7
numIter = 50

# glove = open("/Users/HK/Documents/Courses/2016 Fall/NLP/Final Project/data/glove/glove.6B.200d.txt", "r")
embedSize = 300
glove = open('glove_300.txt', 'rb')

words = pickle.load(glove)


prepList = ['of', 'in', 'for', 'with', 'on']


preSents_train = []
preSentsLen_train = []
postSents_train = []
postSentsLen_train = []

preSents_dev = []
preSentsLen_dev = []
postSents_dev = []
postSentsLen_dev = []

preSents_test = []
preSentsLen_test = []
postSents_test = []
postSentsLen_test = []


train_sentsFile = open("/Users/HK/Documents/Courses/2016 Fall/NLP/Final Project/data/train_sents_2.p", 'rb')
train_labelsFile = open("/Users/HK/Documents/Courses/2016 Fall/NLP/Final Project/data/train_labels_2.p", 'rb')
dev_sentsFile = open("/Users/HK/Documents/Courses/2016 Fall/NLP/Final Project/data/dev_sents_2.p", 'rb')
dev_labelsFile = open("/Users/HK/Documents/Courses/2016 Fall/NLP/Final Project/data/dev_labels_2.p", 'rb')
test_sentsFile = open("/Users/HK/Documents/Courses/2016 Fall/NLP/Final Project/data/test_sents_2.p", 'rb')
test_labelsFile = open("/Users/HK/Documents/Courses/2016 Fall/NLP/Final Project/data/test_labels_2.p", 'rb')

train_Sents = pickle.load(train_sentsFile)
train_labels = pickle.load(train_labelsFile)

dev_Sents = pickle.load(dev_sentsFile)
dev_labels = pickle.load(dev_labelsFile)

test_Sents = pickle.load(test_sentsFile)
test_labels = pickle.load(test_labelsFile)

train_sentsFile.close()
train_labelsFile.close()
dev_sentsFile.close()
dev_labelsFile.close()
test_sentsFile.close()
test_labelsFile.close()

cnt = 0


for sent in train_Sents:

    v = np.where(np.array(sent) == '!@#$%')[0][0]

    preSent = None
    postSent = None

    if v < winSize:
        preSent = sent[0:v]
    else:
        preSent = sent[v-winSize:v]

    if (len(sent) - 1) - v < winSize:
        postSent = sent[v+1:]
    else:
        postSent = sent[v+1:v+1+winSize]

    postSent.reverse()
    
    preSents_train.append(preSent)
    postSents_train.append(postSent)


for sent in dev_Sents:
    v = np.where(np.array(sent) == '!@#$%')[0][0]

    preSent = None
    postSent = None

    if v < winSize:
        preSent = sent[0:v]
    else:
        preSent = sent[v - winSize:v]

    if (len(sent) - 1) - v < winSize:
        postSent = sent[v + 1:]
    else:
        postSent = sent[v + 1:v + 1 + winSize]

    postSent.reverse()

    preSents_dev.append(preSent)
    postSents_dev.append(postSent)


for sent in test_Sents:
    v = np.where(np.array(sent) == '!@#$%')[0][0]

    preSent = None
    postSent = None

    if v < winSize:
        preSent = sent[0:v]
    else:
        preSent = sent[v - winSize:v]

    if (len(sent) - 1) - v < winSize:
        postSent = sent[v + 1:]
    else:
        postSent = sent[v + 1:v + 1 + winSize]

    postSent.reverse()

    preSents_test.append(preSent)
    postSents_test.append(postSent)


trainLen = len(preSents_train)
devLen = len(preSents_dev)
testLen = len(preSents_test)


# words = defaultdict(int)
#
# while True:
#
#     text = glove.readline()
#
#     if text == '':
#         break
#
#     word = text.split()
#     intList = map(float, word[1:])
#
#     words[word[0]] = intList


# pickle.dump(words, out, protocol=2)

X_pre_train = np.zeros((trainLen, winSize, embedSize))
X_pos_train = np.zeros((trainLen, winSize, embedSize))
Y_train = np.zeros((trainLen, 5))

X_pre_dev = np.zeros((devLen, winSize, embedSize))
X_pos_dev = np.zeros((devLen, winSize, embedSize))
Y_dev = np.zeros((devLen, 5))

X_pre_test = np.zeros((testLen, winSize, embedSize))
X_pos_test = np.zeros((testLen, winSize, embedSize))
Y_test = np.zeros((testLen, 5))

k = 0
for i in range(trainLen):

    skip = False

    for j in range(len(preSents_train[i])):

        tempstr = preSents_train[i][j].lower()

        if words[tempstr] == 0:
            skip = True
            X_pre_train[k, :, :] = 0
            break

        X_pre_train[k,j,:] = words[tempstr]

    if skip != True:

        for j in range(len(postSents_train[i])):

            tempstr = postSents_train[i][j].lower()

            if words[tempstr] == 0:
                skip = True
                X_pre_train[k, :, :] = 0
                X_pos_train[k, :, :] = 0
                break

            X_pos_train[k, j, :] = words[tempstr]

    if skip != True:

        Y_train[k,prepList.index(train_labels[i])] = 1

        k += 1



X_pre_train = X_pre_train[:k,:,:]
X_pos_train = X_pos_train[:k,:,:]
Y_train = Y_train[:k,:]


k = 0
for i in range(devLen):

    skip = False

    for j in range(len(preSents_dev[i])):

        tempstr = preSents_dev[i][j].lower()

        if words[tempstr] == 0:

            skip = True
            X_pre_dev[k, :, :] = 0
            break

        X_pre_dev[k,j,:] = words[tempstr]

    if skip != True:

        for j in range(len(postSents_dev[i])):

            tempstr = postSents_dev[i][j].lower()

            if words[tempstr] == 0:
                skip = True
                X_pre_dev[k, :, :] = 0
                X_pos_dev[k, :, :] = 0
                break

            X_pos_dev[k, j, :] = words[tempstr]

    if skip != True:

        Y_dev[k,prepList.index(dev_labels[i])] = 1

        k += 1


X_pre_dev = X_pre_dev[:k,:,:]
X_pos_dev = X_pos_dev[:k,:,:]
Y_dev = Y_dev[:k,:]

k = 0
for i in range(devLen):

    skip = False

    for j in range(len(preSents_test[i])):

        tempstr = preSents_test[i][j].lower()

        if words[tempstr] == 0:

            skip = True
            X_pre_test[k, :, :] = 0
            break

        X_pre_test[k,j,:] = words[tempstr]

    if skip != True:

        for j in range(len(postSents_test[i])):

            tempstr = postSents_test[i][j].lower()

            if words[tempstr] == 0:
                skip = True
                X_pre_test[k, :, :] = 0
                X_pos_test[k, :, :] = 0
                break

            X_pos_test[k, j, :] = words[tempstr]

    if skip != True:

        Y_test[k,prepList.index(test_labels[i])] = 1

        k += 1


out = open("dic.p", 'wb')
pickle.dump(words, out)

X_pre_test = X_pre_test[:k,:,:]
X_pos_test = X_pos_test[:k,:,:]
Y_test = Y_test[:k,:]


S1 = Input(shape=(winSize, embedSize))

lstmPre = LSTM(50, activation = 'tanh', W_regularizer=l2(0.002), return_sequences=False)(S1)

# drop1 = Dropout(0.6)(merged)
dense1 = Dense(30, activation='sigmoid')(lstmPre)
drop2 = Dropout(0.6)(dense1)
dense2 = Dense(5, activation='softmax')(drop2)

model = Model(input=S1, output = dense2)
adm = Adamax(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=adm, metrics=['accuracy', 'fbeta_score'])

model.fit([X_pre_train], Y_train, nb_epoch=numIter, batch_size=200, verbose=1)

print(model.evaluate([X_pre_dev], Y_dev, batch_size=200, verbose=1))
print(model.evaluate([X_pre_test], Y_test, batch_size=200, verbose=1))

