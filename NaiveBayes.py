import nltk
import pickle
from sklearn.metrics import f1_score

# Natural Language Toolkit: code_classification_based_segmenter

def punct_features(tokens, tags, i, labelData):
    nextToken = ''
    nextTag = ''
    secNextTag = ''
    secNextToken = ''
    prevToken = ''
    prevTag = ''
    secPrevToken = ''
    secPrevTag = ''
    bigramPrevWord = ''
    bigramPrevTag = ''
    bigramNextWord =''
    bigramNextTag = ''
    trigramNextWord = ''
    trigramNextTag = ''
    trigramPrevWord = ''
    trigramPrevTag = ''
    if((i+1) < len(tokens)):
        nextToken = tokens[i+1]
    if ((i+2) < len(tokens)):
        secNextToken = tokens[i+2]
        bigramNextWord = tokens[i+1] + " " + tokens[i+2]
    if ((i + 3) < len(tokens)):
        trigramNextWord = tokens[i+1] + " " + tokens[i+2] + " " + tokens[i+3]
    if ((i - 3) < len(tokens)):
        trigramPrevWord = tokens[i-3] + " " + tokens[i-2] + " " + tokens[i-1]
    if ((i + 3) < len(tags)):
        trigramNextTag = tags[i+1] + " " + tags[i+2] + " " + tags[i+3]
    if ((i - 3) < len(tags)):
        trigramPrevTag = tags[i-3] + " " + tags[i-2] + " " + tags[i-1]
    if ((i-1) >= 0):
        prevToken = tokens[i - 1]
    if ((i-2) >= 0):
        secPrevToken = tokens[i - 2]
        bigramPrevWord = tokens[i-2] + " " + tokens[i-1]
    if ((i + 1) < len(tags)):
        nextTag = tags[i + 1]
    if ((i + 2) < len(tags)):
        secNextTag = tags[i + 2]
        bigramNextTag = tags[i + 1] + " " + tags[i + 2]
    if ((i-1) >= 0):
        prevTag = tags[i-1]
    if ((i-2) >= 0):
        secPrevTag = tags[i - 2]
        bigramPrevTag = tags[i - 2] + " " + tags[i - 1]
    return {'next-word': nextToken,
            'second-next-word': secNextToken,
            'next-tag': nextTag,
            'second-next-tag': secNextTag,
            'prev-tag': prevTag,
            'second-prev-tag': secPrevTag,
            'prev-word': prevToken,
            'bigramNextWord': bigramNextWord,
            'bigramPrevWord': bigramPrevWord,
            'bigramNextTag': bigramNextTag,
            'bigramPrevTag': bigramPrevTag,
            'trigramNextWord': trigramNextWord,
            'trigramPrevWord': trigramPrevWord,
            'trigramNextTag': trigramNextTag,
            'trigramPrevTag': trigramPrevTag,
            'second-prev-word':secPrevToken}

tokens = []
tags = []
boundaries = set()
offset = 0
labelData = []
data_train = pickle.load(open("train_sents.p", "rb"))
tags_train = pickle.load(open("train_tags.p", "rb+"))
labels_train = pickle.load(open("train_labels.p", "rb"))
data_dev = pickle.load(open("dev_sents.p", "rb"))
tags_dev = pickle.load(open("dev_tags.p", "rb+"))
labels_dev = pickle.load(open("dev_labels.p", "rb"))

data = data_train + data_dev
labels = labels_train + labels_dev
#tagData = tags_dev + tags_train
#print(tagData)

print(data[0], labels[0])
for i in range(len(data)):
    for word in data[i]:
        tokens.append(word)
        labelData.append(labels[i])

tags = nltk.pos_tag(tokens)
tagData = []
for word, tag in tags:
    if word == '!@#$%':
        tag = 'IN'
        tagData.append(tag)
    else:
        tagData.append(tag)

for i in range(0,len(tokens)):
    if(tokens[i] == '!@#$%'):
        tokens[i] = labelData[i]

#print(tokens)
#prepList = ['aboard', 'about', 'above', 'across', 'after', 'against', 'along', 'amid', 'among', 'anti', 'around', 'as','at', 'before', 'behind', 'below', 'beneath', 'beside', 'besides', 'between', 'beyond', 'but', 'by', 'concerning', 'considering', 'despite', 'down', 'during', 'except', 'excepting', 'excluding', 'following', 'for', 'from', 'in', 'inside', 'into', 'like', 'minus', 'near', 'of', 'off', 'on', 'onto', 'opposite', 'outside', 'over', 'past', 'per', 'plus', 'regarding', 'round', 'save', 'since', 'than', 'through', 'to', 'toward', 'towards', 'under', 'underneath', 'unlike', 'until', 'up', 'upon', 'versus', 'via', 'with', 'within', 'without']
prepList = ['of','in','for','with','on']

featuresets = [(punct_features(tokens, tagData, i, labelData), labelData[i])
                for i in range(0, len(tokens))]

tokens = []
tags=[]
boundaries = set()
offset = 0
labelData = []
data = pickle.load(open("test_sents.p", "rb"))
labels = pickle.load(open("test_labels.p", "rb"))
tagData = pickle.load(open("test_tags.p", "rb+"))

for i in range(len(data)):
    for word in data[i]:
        tokens.append(word)
        labelData.append(labels[i])

tags = nltk.pos_tag(tokens)
tagData = []
for word, tag in tags:
    if word == '!@#$%':
        tag = 'IN'
        tagData.append(tag)
    else:
        tagData.append(tag)

for i in range(0,len(tokens)):
    if(tokens[i] == '!@#$%'):
        tokens[i] = labelData[i]

test_featuresets = [(punct_features(tokens, tagData, i, labelData), labelData[i])
                     for i in range(0, len(tokens))]

train_set = featuresets
test_set = test_featuresets

classifier = nltk.NaiveBayesClassifier.train(train_set)
print(classifier.show_most_informative_features())
print(nltk.classify.accuracy(classifier, test_set))







