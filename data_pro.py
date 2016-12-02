import pickle
from copy import deepcopy
import collections


tr = pickle.load(open("raw_train.p","rb+"))

vocab = collections.OrderedDict()
pos = collections.OrderedDict()
labels = collections.OrderedDict()

label_list = ['aboard', 'about', 'above', 'across', 'after', 'against', 'along', 'amid', 'among', 'anti', 'around', 'as', 'at', 'before', 'behind', 'below', 'beneath', 'beside', 'besides', 'between', 'beyond', 'but', 'by', 'concerning', 'considering', 'despite', 'down', 'during', 'except', 'excepting', 'excluding', 'following', 'for', 'from', 'in', 'inside', 'into', 'like', 'minus', 'near', 'of', 'off', 'on', 'onto', 'opposite', 'outside', 'over', 'past', 'per', 'plus', 'regarding', 'round', 'save', 'since', 'than', 'through', 'to', 'toward', 'towards', 'under', 'underneath', 'unlike', 'until', 'up', 'upon', 'versus', 'via', 'with', 'within', 'without']

for i in label_list:
	if(i not in labels):
		labels[i] = 0

for i in tr:
	for j in i:
		if(j[0].lower() not in vocab):
			vocab[j[0]] = 0
		if(j[1] not in pos):
			pos[j[1]] = 0

print("label_list and tr done")

vocab['!@#$%'] = 0
pos['!@#$%'] = 0

vocab['!@#$'] = 0
pos['!@#$'] = 0

tr_sents = []
tr_pos = []

for i in tr:
	temp_sent = []
	temp_pos = []
	for j in i:
		temp_sent.append(deepcopy(j[0]))
		temp_pos.append(deepcopy(j[1]))

	tr_sents.append(deepcopy(temp_sent))
	tr_pos.append(deepcopy(temp_pos))

print("tr lists done")
actual_sents = []
actual_labels = []
for i in range(0,len(tr_sents)):
	for j in range(0,len(tr_sents[i])):
		if(tr_pos[i][j] == "IN"):
			if(tr_sents[i][j].lower() not in labels):
				labels[tr_sents[i][j].lower()] = 0
				label_list.append(tr_sents[i][j].lower())
			tempo = deepcopy(tr_sents[i])
			tempo[j] = "!@#$%"
			actual_sents.append(deepcopy(tempo))
			actual_labels.append(deepcopy(tr_sents[i][j].lower()))

print("actual lists done")




pickle.dump(vocab,open("vocab.p","wb+"))
pickle.dump(pos,open("pos.p","wb+"))
pickle.dump(labels,open("labels.p","wb+"))
pickle.dump(label_list,open("label_list.p","wb+"))

pickle.dump(actual_sents,open("actual_sents.p","wb+"))
pickle.dump(actual_labels,open("actual_labels.p","wb+"))
pickle.dump(tr_sents,open("tr_sents.p","wb+"))
pickle.dump(tr_pos,open("tr_pos.p","wb+"))

print("all done")