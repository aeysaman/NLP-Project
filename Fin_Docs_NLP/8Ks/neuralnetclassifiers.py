## 4/26/18

import pickle
from nltk import FreqDist
from nltk.corpus import stopwords
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn import metrics

def score(change):
    if change <-.02:
        return "down"
    elif change <.02:
        return "flat"
    else:
        return "up"

stops = stopwords.words('english')
stops.extend([",", ".", "!", "?", "'", '"', "I", "i", "n't", "'ve", "'d", "'s"])

## separate down, flat, up docs
allwords = [] 
down = []
flat = []
up = []

with open("C:/Users/aldoj/Documents/Natural Language Processing/Final Project/8Ks - Cons Disc 10B.p", "rb") as f:
    docs = pickle.load(f)
docs = [d for d in docs if not d["2-2dayPriceChange"] is None]

for d in docs:
    tokens = list(set([w for w in d["tokens"] if not w in stops]))
    change = score(d["2-2dayPriceChange"])
    if change=="down":
        down.append(tokens)
    elif change=="flat":
        flat.append(tokens)
    else:
        up.append(tokens)
    allwords.extend(tokens)

## separate training & testing data
trainingdown = down[:300]
trainingflat = flat[:300]
trainingup = up[:300]

testingdown = down[300:]
testingflat = flat[300:]
testingup = up[300:]

## get 1,000 most frequent words
wfreq = FreqDist(allwords)
top1000 = wfreq.most_common(1000)

## format training data
training = []
traininglabel = []

for d in trainingdown:
    vec = []
    for t in top1000:
        if t[0] in d:
            vec.append(1)
        else:
            vec.append(0)
    training.append(vec)
    traininglabel.append("down")

for f in trainingflat:
    vec = []
    for t in top1000:
        if t[0] in f:
            vec.append(1)
        else:
            vec.append(0)
    training.append(vec)
    traininglabel.append("flat")

for u in trainingup:
    vec = []
    for t in top1000:
        if t[0] in u:
            vec.append(1)
        else:
            vec.append(0)
    training.append(vec)
    traininglabel.append("up")

## format testing data
testing = []
testinglabel = []

for d in testingdown:
    vec = []
    for t in top1000:
        if t[0] in d:
            vec.append(1)
        else:
            vec.append(0)
    testing.append(vec)
    testinglabel.append("down")

for f in testingflat:
    vec = []
    for t in top1000:
        if t[0] in f:
            vec.append(1)
        else:
            vec.append(0)
    testing.append(vec)
    testinglabel.append("flat")

for u in testingup:
    vec = []
    for t in top1000:
        if t[0] in u:
            vec.append(1)
        else:
            vec.append(0)
    testing.append(vec)
    testinglabel.append("up")

## MLP
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(training, traininglabel)
predicted = clf.predict(testing)
print("Accruracy of Multi-Layer Perceptron")
print(metrics.classification_report(testinglabel, predicted))

## SVC
svcclf = LinearSVC()
svcclf.fit(training, traininglabel)
predicted = svcclf.predict(testing)
print("Accruracy of LinearSVC")
print(metrics.classification_report(testinglabel, predicted))


## Gaussian
nbclf = GaussianNB()
nbclf.fit(training, traininglabel)
predicted = nbclf.predict(testing)
print("Accruracy of GaussianNB")
print(metrics.classification_report(testinglabel, predicted))


