'''
Created on Apr 1, 2018

@author: aldoj
'''

import pickle

from nltk import FreqDist
from nltk.stem.wordnet import WordNetLemmatizer
from numpy import median, mean
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.svm.classes import LinearSVC

period = "-2+2dayPriceChange"

## read in data from lexicon file
data = {} ## string(word):tuple(valence,arousal,dominance)
f = open("lexicon.csv")
next(f) ## skips title row of csv file
for line in f:
    strArray = line[0:len(line)-1].split(",")
    values = (float(strArray[1]),float(strArray[2]),float(strArray[3]))
    data[strArray[0]] = values
f.close()

fnames = ["8Ks_Utils_10B.p"]
print("File: %s"% " ".join(fnames))
#read Data from prepped file
def getPrepped(name):
    with open("C:/Users/aldoj/Documents/Natural Language Processing/Final Project/"+name, "rb") as f:
        return pickle.load(f)
docs = [doc for f in fnames for doc in getPrepped(f)]
    
#create training & test data
docs = [d for d in docs if not d[period] is None]
print(len(docs))
    
years = [d["date"].year for d in docs]

year = 2008
training = [d for d in docs if d["date"].year <=year]
testing = [d for d in docs if d["date"].year >=year]

print("Training: %d\tTesting: %d" %(len(training), len(testing)))

def score(change):
    if change <-.02:
        return "down"
    elif change <.02:
        return "flat"
    else:
        return "up"

#Generate Features & Results 
# train_data = [(FreqDist(d["tokens"]), score(d[period])) for d in training]
# test_features = [FreqDist(d["tokens"]) for d in testing]
# print("done with features")

train_results = [score(d[period]) for d in training]
test_results = [score(d[period]) for d in testing]
print("done with results")
 
#Frequency Labels
wfreq = FreqDist([x for d in training for x in d["tokens"]])
n = 5000
topwords = [word for word, freq in wfreq.most_common(n)]
 
def topCounts(toks):
    freq = FreqDist(toks)
    return [freq[word] if word in freq else 0 for word in topwords]

# numOf = lambda x, ls: sum([x==foo for foo in ls])
 
train_freq = [topCounts(doc["tokens"]) for doc in training]
test_freq = [topCounts(doc["tokens"]) for doc in testing]
# test_freq = [[numOf(term, doc["tokens"]) for term in topwords] for doc in testing]

print("done with freq")

#Affective Labels
lemmatizer = WordNetLemmatizer()

affectscores = lambda x: [data[w] for w in x["tokens"] if w in data]

train_affect = [affectscores(x) for x in training]
test_affect = [affectscores(x) for x in testing]
  
fns = (sum, max, min, median, mean)

affectfeatures = lambda ls: [ [f([foo[i] for foo in x ]) for f in fns for i in range(3)] if len(x) >0 else [0]*3*len(fns) for x in ls]
train_affect_features = affectfeatures(train_affect)
test_affect_features = affectfeatures(test_affect)

print("done with affect")

#Print Results
amounts = [ (direction, len([ t for t in test_results if t ==direction])) for direction in ["down", "flat", "up"]]
print(amounts)
print("Majority Baseline: %.2f" % (max([b for a,b in amounts]) / len(test_results)))
 
MLP = ("Multi-Layer Perceptron", MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1))
SVM = ("LinearSVC", LinearSVC())
GNB = ("Gaussian Naive Bayes", GaussianNB())
 
print("With %d most common terms" % n)

for name, m in (MLP, GNB, SVM):
    m.fit(train_freq, train_results)
    pred = m.predict(test_freq)
    print(name)
    print(metrics.classification_report(test_results, pred))

print("With Affect Features")

for name, m in (MLP, GNB, SVM):
    m.fit(train_affect_features, train_results)
    pred = m.predict(test_affect_features)
    print(name)
    print(metrics.classification_report(test_results, pred))


# 
# #Train Model
# model = nltk.NaiveBayesClassifier.train(train_data)
#  
# #Generate Predictions
# preds = model.classify_many(test_features)
#  
# print("Accuracy: %.2f" % (nltk.accuracy(preds, test_results)))
# 
# print(ConfusionMatrix(preds, test_results))
#  
# print(model.show_most_informative_features(10))

