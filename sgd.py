import pickle
from copy import deepcopy
import numpy as np
from sklearn.linear_model import SGDClassifier
import pycrfsuite

tr_list = pickle.load(open("train_new_feats.p","rb+"))
tr_labels = pickle.load(open("train_labs.p","rb+"))

de_list = pickle.load(open("dev_new_feats.p","rb+"))
de_labels = pickle.load(open("dev_labs.p","rb+"))

te_list = pickle.load(open("test_new_feats.p","rb+"))
te_labels = pickle.load(open("test_labs.p","rb+"))



learning = [0.0001,0.01,0.1,1,10,100,1000]
best_acc = 0
best_cla = 0
best_c = 0
loss = [ 'str','hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron','squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
for i in learning:
	for j in loss:
		cla = SGDClassifier(learning_rate=i,loss=j)

		cla.fit(tr_list,tr_labels)

		res = list(cla.predict(de_list))

		correct = 0

		for i in range(0,len(res)):
			if(res[i] == (de_labels[i])):
				correct += 1
		if((correct/len(res)) > best_acc):
			best_cla = deepcopy(cla)
			best_c = i


res = list(best_cla.predict(te_list))

correct = 0

for i in range(0,len(res)):
	if(res[i] == (te_labels[i])):
		correct += 1

print(correct/len(res))