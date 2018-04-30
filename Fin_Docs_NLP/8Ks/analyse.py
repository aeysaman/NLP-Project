'''
Created on Apr 1, 2018

@author: aldoj
'''

from itertools import count
import pickle

from nltk import ConfusionMatrix
from nltk import FreqDist
import nltk
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.svm.classes import LinearSVC


#read Data from prepped file
with open("C:/Users/aldoj/Documents/Natural Language Processing/Final Project/8Ks - Cons Disc 10B.p", "rb") as f:
    docs = pickle.load(f)
    
#create training & test data
docs = [d for d in docs if not d["2-2dayPriceChange"] is None]
print(len(docs))
    
training = docs[:1500]
testing = docs[1500:]

def score(change):
    if change <-.02:
        return "down"
    elif change <.02:
        return "flat"
    else:
        return "up"

#Generate Features & Results 
train_data = [(FreqDist(d["tokens"]), score(d["2-2dayPriceChange"])) for d in training]
train_results = [score(d["2-2dayPriceChange"]) for d in training]
test_features = [FreqDist(d["tokens"]) for d in testing]
test_results = [score(d["2-2dayPriceChange"]) for d in testing]

#Frequency Labels
wfreq = FreqDist([x for d in training + testing for x in d["tokens"]])
topwords = [word for word, freq in wfreq.most_common(1000)]


numOf = lambda x, ls: sum([x==foo for foo in ls])

train_freq = [[numOf(term, doc["tokens"]) for term in topwords] for doc in training]
test_freq = [[numOf(term, doc["tokens"]) for term in topwords] for doc in testing]

# 
# #Train Model
# model = nltk.NaiveBayesClassifier.train(train_data)
#  
# #Generate Predictions
# preds = model.classify_many(test_features)

#Print Results
amounts = [ (direction, len([ t for t in test_results if t ==direction])) for direction in ["down", "flat", "up"]]
print(amounts)
print("Majority Baseline: %.2f" % (max([b for a,b in amounts]) / len(test_results)))

MLP = ("Multi-Layer Perceptron", MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1))
SVM = ("LinearSVC", LinearSVC())
GNB = ("Gaussian Naive Bayes", GaussianNB())

for name, m in (MLP, GNB, SVM):
    m.fit(train_freq, train_results)
    pred = m.predict(test_freq)
    print(name)
    print(metrics.classification_report(test_results, pred))

# print("Accuracy: %.2f" % (nltk.accuracy(preds, test_results)))
# 
# print(ConfusionMatrix(preds, test_results))
#  
# print(model.show_most_informative_features(10))

