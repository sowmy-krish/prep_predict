import pickle


test_labs = pickle.load(open("dev_labs.p","rb+"))
label_list = ['of','in','for','with','on']

labs = []

for i in test_labs:
	labs.append(label_list.index(i))

pickle.dump(labs,open("dev_labs.p","wb+"))