import pickle
import pycrfsuite
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
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
print("starting classifier..")
clf = RandomForestClassifier(n_estimators=20)
param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 5),
              "min_samples_split": sp_randint(1, 5),
              "min_samples_leaf": sp_randint(1, 5),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
n_iter_search = 10
random_search = RandomizedSearchCV(clf,   param_distributions=param_dist, n_iter=n_iter_search)
print("Fitting the model..")
random_search.fit(X_data, y_data)
print("Model fitting done..")
#validation_score = random_search.score(X_valid, y_valid)
#random_search.fit(X_model, y_model)

y_pred = random_search.predict(X_test)
print("Random forest output: ", random_search.score(X_test, y_test))

f1Score = metrics.f1_score(y_test, y_pred,labels=None,pos_label=1,average='weighted')
Precision = metrics.precision_score(y_test,y_pred,labels=None,pos_label=1,average='weighted')
Recall = metrics.recall_score(y_test, y_pred,labels=None,pos_label=1,average='weighted')

print("Random Forest Precision, Recall and F1 Score:", Precision, Recall, f1Score)

report = classification_report(y_test,y_pred)
print(report)


