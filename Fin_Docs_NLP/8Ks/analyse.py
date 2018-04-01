'''
Created on Apr 1, 2018

@author: aldoj
'''

import pickle
import nltk
from nltk import FreqDist
from nltk import ConfusionMatrix

with open("8Ks_&_prices.p", "rb") as f:
        docs = pickle.load(f)

def score(change):
    if change <-.03:
        return "down"
    elif change <.03:
        return "flat"
    else:
        return "up"
    
    
docs = [d for d in docs if not d["2-2dayPriceChange"] is None]
print(len(docs))
    
training = docs[:500]
testing = docs[500:]
    
training_data = [(FreqDist(d["tokens"]), score(d["2-2dayPriceChange"])) for d in training]
test_features = [FreqDist(d["tokens"]) for d in testing]
test_results = [score(d["2-2dayPriceChange"]) for d in testing]
    
model = nltk.NaiveBayesClassifier.train(training_data)

preds = model.classify_many(test_features)
print(ConfusionMatrix(preds, test_results))

print(model.show_most_informative_features(10))