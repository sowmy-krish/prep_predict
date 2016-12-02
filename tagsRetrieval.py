import nltk
import pickle

tags = []
taggedData = []
words = []
tagList = []
tagged = []
data = pickle.load(open("train_sents.p", "rb"))
#data = pickle.load(open("dev_sents.p", "rb"))
#data = pickle.load(open("test_sents.p", "rb"))
for i in range(len(data)):
    tags = nltk.pos_tag(data[i])
    tagList.append(tags)


for i in range(0,len(tagList)):
    tagData = []
    print(tagList[i])
    for word, tag in tagList[i]:
        if word == '!@#$%':
            tag = 'IN'
            tagData.append(tag)
        else:
            tagData.append(tag)
    tagged.append(tagData)

print(tagged[0])
pickle.dump(tagged, open("train_tags.p","wb"))
#pickle.dump(tagged, open("dev_tags.p","wb"))
#pickle.dump(tagged, open("test_tags.p","wb"))