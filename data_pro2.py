import pickle
from copy import deepcopy

actual_sents = pickle.load(open("actual_sents.p","rb+"))
actual_labels = pickle.load(open("actual_labels.p","rb+"))

vocab = pickle.load(open("vocab.p","rb+"))

label_list = ["of","in","for","with","on"]

train_indices = pickle.load(open("train_indices.p","rb+"))
dev_indices = pickle.load(open("dev_indices.p","rb+"))
test_indices = pickle.load(open("test_indices.p","rb+"))


feats = []
count = 1

train_sents = []
dev_sents = []
test_sents = []

train_labs = []
dev_labs = []
test_labs = []


for i in train_indices:
	train_sents.append(actual_sents[i])
	train_labs.append(actual_labels[i])

for i in dev_indices:
	dev_sents.append(actual_sents[i])
	dev_labs.append(actual_labels[i])

for i in test_indices:
	test_sents.append(actual_sents[i])
	test_labs.append(actual_labels[i])


# print(train_sents)
# print(train_labs)
#
# print(dev_sents)
# print(dev_labs)
#
# print(test_sents)
# print(test_labs)


pickle.dump(train_sents,open("train_sents.p","wb+"))
pickle.dump(dev_sents,open("dev_sents.p","wb+"))
pickle.dump(test_sents,open("test_sents.p","wb+"))

pickle.dump(train_labs,open("train_labels.p","wb+"))
pickle.dump(dev_labs,open("dev_labels.p","wb+"))
pickle.dump(test_labs,open("test_labels.p","wb+"))

pickle.dump(label_list,open("lab_list.p","wb+"))

for i in test_sents:
	print(count)
	temp_d = deepcopy(vocab)
	for j in i:
		if(j.lower() not in temp_d):
			temp_d['!@#$'] += 1
		else:
			temp_d[j.lower()] +=1
	feats.append(list(temp_d.values()))
	count+=1
print("feats lists done")

labs = []

for i in test_labs:
	labs.append(label_list.index(i))


print("labs lists done")

print(feats)
print(labs)

pickle.dump(feats,open("test_feats.p","wb+"))
pickle.dump(labs,open("test_labs.p","wb+"))
