import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score

print('#50')
print('Loading...')
newscorpora = pd.read_csv('newsCorpora.csv',sep='\t',header=None)
newscorpora.columns = ['ID','TITLE','URL','PUBLISHER','CATEGORY','STORY','HOSTNAME','TIMESTAMP']
publisher = ['Reuters', 'Huffington Post', 'Businessweek', 'Contactmusic.com', 'Daily Mail']
df=newscorpora[newscorpora['PUBLISHER'].isin(publisher)]

df = df.sample(frac=1)

train_df, valid_test_df = train_test_split(df, test_size=0.2)
valid_df, test_df = train_test_split(valid_test_df, test_size=0.5)
train_df.to_csv('train.txt', columns = ['CATEGORY','TITLE'], sep='\t',header=False, index=False)
valid_df.to_csv('valid.txt', columns = ['CATEGORY','TITLE'], sep='\t',header=False, index=False)
test_df.to_csv('test.txt', columns = ['CATEGORY','TITLE'], sep='\t',header=False, index=False)

print(df['CATEGORY'].value_counts())

print('#51')
print('Vectorizing...')
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
x_train = vectorizer.fit_transform(train_df['TITLE'])
x_valid = vectorizer.transform(valid_df['TITLE'])
x_test = vectorizer.transform(test_df['TITLE'])
y_train = train_df['CATEGORY']
y_valid = valid_df['CATEGORY']
y_test = test_df['CATEGORY']
print(x_train.shape)
print(x_valid.shape)
print(x_test.shape)

print('Selecting features...')
selector = SelectKBest(k=7000, score_func=mutual_info_classif)
selector.fit(x_train, y_train)
x_train = selector.transform(x_train)
x_valid = selector.transform(x_valid)
x_test = selector.transform(x_test)
print(x_train.shape)
print(x_valid.shape)
print(x_test.shape)

print('Saving...')
np.savetxt('train.feature.txt', x_train.toarray(), fmt='%f')
np.savetxt('valid.feature.txt', x_valid.toarray(), fmt='%f')
np.savetxt('test.feature.txt', x_test.toarray(), fmt='%f')

print('#52')
print('Logistic...')
clf = LogisticRegression(max_iter=4000)
clf.fit(x_train, y_train)

print('#53')
dic = {'b':'business', 't':'science and technology', 'e' : 'entertainment', 'm' : 'health'}

def predict(text):
    text = [text]
    x = vectorizer.transform(text)
    x = selector.transform(x)
    list_proba = clf.predict_proba(x)
    for proba in list_proba:
        for c, p in zip(clf.classes_, proba):
            print (dic[c]+':',p)

s = 'Nasa and Boeing sign $2.8bn deal to build rocket to take us to Mars'
print(s)
predict(s)

print('#54')
y_train_pred = clf.predict(x_train)
y_test_pred = clf.predict(x_test)
print (accuracy_score(y_train, y_train_pred))
print (accuracy_score(y_test, y_test_pred))

print('#55')
print('Train matrix')
print (confusion_matrix(y_train, y_train_pred, labels=['b','t','e','m']))
print('Test matrix')
print (confusion_matrix(y_test, y_test_pred, labels=['b','t','e','m']))

print('#56')
print('None')
print(precision_score(y_test, y_test_pred, average=None, labels=['b','t','e','m']))
print(recall_score(y_test, y_test_pred, average=None, labels=['b','t','e','m']))
print(f1_score(y_test, y_test_pred, average=None, labels=['b','t','e','m']))
print('Micro')
print(precision_score(y_test, y_test_pred, average='micro', labels=['b','t','e','m']))
print(recall_score(y_test, y_test_pred, average='micro', labels=['b','t','e','m']))
print(f1_score(y_test, y_test_pred, average='micro', labels=['b','t','e','m']))
print('Macro')
print(precision_score(y_test, y_test_pred, average='macro', labels=['b','t','e','m']))
print(recall_score(y_test, y_test_pred, average='macro', labels=['b','t','e','m']))
print(f1_score(y_test, y_test_pred, average='macro', labels=['b','t','e','m']))

print('#57')
names = np.array(vectorizer.get_feature_names())
labels=['b','t','e','m']
for c, coef in zip(clf.classes_, clf.coef_):
    idx = np.argsort(coef)[::-1]
    print (dic[c])
    print (names[idx][:10])
    print (names[idx][-10:][::-1])

print('#58')
def calc_scores(c):

    clf = LogisticRegression(C=c,max_iter=4000)
    clf.fit(x_train, y_train)

    y_train_pred = clf.predict(x_train)
    y_valid_pred = clf.predict(x_valid)
    y_test_pred = clf.predict(x_test)

    scores = []
    scores.append(accuracy_score(y_train, y_train_pred))
    scores.append(accuracy_score(y_valid, y_valid_pred))
    scores.append(accuracy_score(y_test, y_test_pred))
    return scores

C = np.logspace(-5, 5,num=11, base=10)
scores = []
for c in C:
    scores.append(calc_scores(c))
scores = np.array(scores).T
labels = ['train', 'valid', 'test']

for score, label in zip(scores,labels):
    plt.plot(C, score, label=label)
plt.ylim(0, 1.1)
plt.xscale('log')
plt.xlabel('C', fontsize = 14)
plt.ylabel('Accuracy', fontsize = 14)
plt.tick_params(labelsize=14)
plt.grid(True)
plt.legend()
plt.show()

print('#59')
def calc_scores(c,solver,class_weight):

    clf = LogisticRegression(C=c, solver=solver, class_weight=class_weight,max_iter=4000)
    clf.fit(x_train, y_train)

    y_train_pred = clf.predict(x_train)
    y_valid_pred = clf.predict(x_valid)
    y_test_pred = clf.predict(x_test)

    scores = []
    scores.append(accuracy_score(y_train, y_train_pred))
    scores.append(accuracy_score(y_valid, y_valid_pred))
    scores.append(accuracy_score(y_test, y_test_pred))
    return scores

C = np.logspace(-5, 5,num=11, base=10)
solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
class_weight = [None, 'balanced']
best_parameter = None
best_scores = None
max_valid_score = 0
for c, s, w in itertools.product(C, solver, class_weight):
    scores = calc_scores(c, s, w)
    if scores[1] > max_valid_score:
        max_valid_score = scores[1]
        best_parameter = [c, s, w]
        best_scores = scores
print('Best parameter: ', best_parameter)
print('Score: ', best_scores)
