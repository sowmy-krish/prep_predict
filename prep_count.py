import pickle
from copy import deepcopy
import collections

actual_sents = pickle.load(open("actual_sents.p","rb+"))
actual_labels = pickle.load(open("actual_labels.p","rb+"))
label_list = pickle.load(open("label_list.p","rb+"))
dic = pickle.load(open("dic.p","rb+"),encoding='UTF-8')



labels = {}

labels["of"] = 0
labels["in"] = 0
labels["for"] = 0
labels["with"] = 0
labels["on"] = 0

train_indices = []
dev_indices = []
test_indices = []
print((dic.keys()))
for i in range(0, len(actual_labels)):
	if((actual_labels[i] =="of" or actual_labels[i] =="in" or actual_labels[i] =="for" or actual_labels[i] =="with" or actual_labels[i] =="on") and (len(actual_sents[i]) <= 30)):
		flag = 0
		for j in actual_sents[i]:
			if(j.isalpha()):
				if(j.lower() not in dic):
					flag =1
		if(flag == 0):
			if(labels[actual_labels[i]] <= 600):
				train_indices.append(i)
			elif(labels[actual_labels[i]] > 600 and labels[actual_labels[i]] <=900):
				dev_indices.append(i)
			elif(labels[actual_labels[i]] > 900 and labels[actual_labels[i]] <=1200):
				test_indices.append(i)
			labels[actual_labels[i]]+=1

print(len(train_indices))
print(len(dev_indices))
print(len(test_indices))
pickle.dump(train_indices,open("train_indices.p","wb+"))
pickle.dump(dev_indices,open("dev_indices.p","wb+"))
pickle.dump(test_indices,open("test_indices.p","wb+"))
