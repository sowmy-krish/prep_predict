import pickle
import pycrfsuite
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.metrics import classification_report

def punct_features(tokens, tags, i, labelData):
    features = []
    for j in tokens[i]:
        if j == '!@#$%':
            features.append("TOKEN_" + labelData[i])
        else:
            features.append("TOKEN_" + j)
    for j in tags[i]:
        features.append("TAG_"+ j)
    prevWord = ''
    nextWord = ''
    prevToken = ''
    nextToken = ''
    bigramPrevWord = ''
    bigramNextWord = ''
    bigramNextTag = ''
    bigramPrevTag = ''

    for j in range(0,len(tokens[i])):
        if tokens[i][j] == '!@#$%':
            position = j
            features.append(str(position))
            tokens[i][j] = labelData[i]
    prevWordlist = []
    prevTagList = []
    for j in range(0, len(tokens[i])):
        if(j == position):
            if j-1 >= 0:
                prevWord = "PREVTOKEN_" + tokens[i][j-1]
                prevToken = "PREVTAG_" + tags[i][j-1]
                prevWordlist.append(prevWord)
                prevTagList.append(prevToken)
            else:
                prevWord = "PREVTOKEN_"
                prevToken = "PREVTAG_"
                prevWordlist.append(prevWord)
                prevTagList.append(prevToken)
    features.extend(prevWordlist)
    features.extend(prevTagList)

    bigramPrevWordList = []
    bigramPrevtagList = []
    for j in range(0, len(tokens[i])):
        if(j == position):
            if j-2 >= 0:
                bigramPrevWord = "BIGRAMPREVTOK_" + tokens[i][j-2] + " " + tokens[i][j-1]
                bigramPrevTag = "BIGRAMPREVTAG_" + tags[i][j-2] + " " + tags[i][j-1]
                bigramPrevWordList.append(bigramPrevWord)
                bigramPrevtagList.append(bigramPrevTag)
            else:
                bigramPrevWord = "BIGRAMPREVTOK_"
                bigramPrevTag = "BIGRAMPREVTAG_"
                bigramPrevWordList.append(bigramPrevWord)
                bigramPrevtagList.append(bigramPrevTag)
    features.extend(bigramPrevWordList)
    features.extend(bigramPrevtagList)

    bigramNextWordList = []
    bigramNextTagList = []
    for j in range(0, len(tokens[i])):
        if(j == position):
            if j+2 < len(tokens[i]):
                bigramNextWord = "BIGRAMNEXTTOK_" + tokens[i][j+1] + " " + tokens[i][j+2]
                bigramNextTag = "BIGRAMNEXTTAG_" + tags[i][j+1] + " " + tags[i][j+2]
                bigramNextWordList.append(bigramNextWord)
                bigramNextTagList.append(bigramNextTag)
            else:
                bigramNextWord = "BIGRAMNEXTTOK_"
                bigramNextTag = "BIGRAMNEXTTAG_"
                bigramNextWordList.append(bigramNextWord)
                bigramNextTagList.append(bigramNextTag)
    features.extend(bigramNextWordList)
    features.extend(bigramNextTagList)

    nextWordList = []
    nextTokenList = []
    for j in range(0, len(tokens[i])):
        if(j == position):
            if j+1 < len(tokens[i]):
                nextWord = "NEXTWORD_" + tokens[i][j+1]
                nextToken = "NEXTTAG_" + tags[i][j+1]
                nextWordList.append(nextWord)
                nextTokenList.append(nextToken)
            else:
                nextWord = "NEXTWORD_"
                nextToken = "NEXTTAG_"
                nextWordList.append(nextWord)
                nextTokenList.append(nextToken)

    features.extend(nextWordList)
    features.extend(nextTokenList)

    return features

tr_list = pickle.load(open("train_sents.p","rb+"))
tr_tags = pickle.load(open("train_tags.p", "rb+"))
tr_labels = pickle.load(open("train_labels.p","rb+"))

de_list = pickle.load(open("dev_sents.p","rb+"))
de_tags = pickle.load(open("dev_tags.p", "rb+"))
de_labels = pickle.load(open("dev_labels.p","rb+"))

te_list = pickle.load(open("test_sents.p","rb+"))
te_tags = pickle.load(open("test_tags.p", "rb+"))
te_labels = pickle.load(open("test_labels.p","rb+"))

print(tr_list[0])
trainer = pycrfsuite.Trainer(verbose=False)
tokens = tr_list + de_list
tags = tr_tags + de_tags
labelData = tr_labels + de_labels

prepList = ['of','in','for','with','on']

print(len(tokens), len(tags))
train_featuresets = [punct_features(tokens, tags, i, labelData) for i in range(0, len(tokens))]

X_train_list = []
y_train_list = []
X_train = train_featuresets
y_train = labelData

for i in range(0,len(X_train)):
    X_train_list.append([X_train[i]])
    y_train_list.append([y_train[i]])


crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=70,
    all_possible_transitions=True)

crf.fit(X_train_list,y_train_list)

tokens = te_list
tags = te_tags
labelData = te_labels
test_featuresets = [punct_features(tokens, tags, i, labelData) for i in range(0, len(tokens))]

X_test = test_featuresets
y_test = labelData
y_pred = crf.predict([X_test])


f1_Score = metrics.flat_f1_score([y_test],y_pred,average='weighted')
precision = metrics.flat_precision_score([y_test], y_pred, average='weighted')
accuracy = metrics.flat_accuracy_score([y_test],y_pred)
print("F1 Score :" , f1_Score, "Precision :" , precision, "Accuracy :" , accuracy)
