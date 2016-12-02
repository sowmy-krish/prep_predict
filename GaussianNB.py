import pickle
import pycrfsuite
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import classification_report


tr_list = pickle.load(open("train_feats.p","rb+"))
tr_vocab = pickle.load(open("train_labs.p","rb+"))

de_list = pickle.load(open("dev_feats.p","rb+"))
de_vocab = pickle.load(open("dev_labs.p","rb+"))

te_list = pickle.load(open("test_feats.p","rb+"))
te_vocab = pickle.load(open("test_labs.p","rb+"))

X_data = tr_list + de_list
y_data = tr_vocab + de_vocab

X_valid = de_list
y_valid = de_vocab

X_test = te_list
y_test = te_vocab

best_score_train = 0
best_score_validation = 0
validation_score = 0

gnb = GaussianNB()
print(len(X_data), len(y_data))
gnb.fit(X_data, y_data)

y_pred = gnb.predict(X_test)

f1Score = metrics.f1_score(y_test, y_pred,labels=None,pos_label=1,average='weighted')
Precision = metrics.precision_score(y_test,y_pred,labels=None,pos_label=1,average='weighted')
Recall = metrics.recall_score(y_test, y_pred,labels=None,pos_label=1,average='weighted')

print("Gaussian NB Precision, Recall and F1 Score:", Precision, Recall, f1Score)

report = classification_report(y_test,y_pred)
print(report)



