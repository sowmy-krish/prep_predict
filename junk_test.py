import pickle
from copy import deepcopy

voc_list = pickle.load(open("voc_list.p","rb+"))

tr_list = pickle.load(open("train_sents.p","rb+"))
de_list = pickle.load(open("dev_sents.p","rb+"))
te_list = pickle.load(open("test_sents.p","rb+"))


tr_new_feats = []
dev_new_feats = []
test_new_feats = []

for i in tr_list:
	temp_list = []
	for j in i:
		if(j in voc_list):
			temp_list.append(voc_list.index(j))
		else:
			temp_list.append(voc_list.index("!@#$"))
	for k in range(0,30 - len(i)):
		temp_list.append(-1)

	tr_new_feats.append(deepcopy(temp_list))

for i in de_list:
	temp_list = []
	for j in i:
		if(j in voc_list):
			temp_list.append(voc_list.index(j))
		else:
			temp_list.append(voc_list.index("!@#$"))
	for k in range(0,30 - len(i)):
		temp_list.append(-1)

	dev_new_feats.append(deepcopy(temp_list))

for i in te_list:
	temp_list = []
	for j in i:
		if(j in voc_list):
			temp_list.append(voc_list.index(j))
		else:
			temp_list.append(voc_list.index("!@#$"))
	for k in range(0,30 - len(i)):
		temp_list.append(-1)

	test_new_feats.append(deepcopy(temp_list))

for i in tr_new_feats:
	print(i)

pickle.dump(tr_new_feats,open("train_new_feats.p","wb+"))
pickle.dump(dev_new_feats,open("dev_new_feats.p","wb+"))
pickle.dump(test_new_feats,open("test_new_feats.p","wb+"))
