'''
Created on Apr 1, 2018

@author: aldoj
'''

import pickle
import nltk
from nltk import FreqDist
from nltk import ConfusionMatrix

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
training_data = [(FreqDist(d["tokens"]), score(d["2-2dayPriceChange"])) for d in training]
test_features = [FreqDist(d["tokens"]) for d in testing]
test_results = [score(d["2-2dayPriceChange"]) for d in testing]

#Train Model
model = nltk.NaiveBayesClassifier.train(training_data)
 
#Generate Predictions
preds = model.classify_many(test_features)

#Print Results
amounts = [ (direction, len([ t for t in test_results if t ==direction])) for direction in ["down", "flat", "up"]]
print(amounts)
print("Majority Baseline: %.2f" % (max([b for a,b in amounts]) / len(test_results)))

print("Accuracy: %.2f" % (nltk.accuracy(preds, test_results)))

print(ConfusionMatrix(preds, test_results))
 
print(model.show_most_informative_features(10))

