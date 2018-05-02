## 5/1/2018

import pickle
import numpy as np
from nltk import FreqDist
from nltk.stem import WordNetLemmatizer
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

## read in data from lexicon file
data = {} ## string(word):tuple(valence,arousal,dominance)
f = open("lexicon.csv")
next(f) ## skips title row of csv file
for line in f:
    strArray = line[0:len(line)-1].split(",")
    values = (float(strArray[1]),float(strArray[2]),float(strArray[3]))
    data[strArray[0]] = values
f.close()

## read Data from prepped file
with open("8Ks - Cons Disc 10B.p", "rb") as f:
    docs = pickle.load(f)
    
## create training & test data
docs = [d for d in docs if not d["2-2dayPriceChange"] is None]
    
training = docs[:500]
testing = docs[500:]

def score(change):
    if change <-.02:
        return "down"
    elif change <.02:
        return "flat"
    else:
        return "up"

## format training & testinf data
training_data = [(FreqDist(d["tokens"]), score(d["2-2dayPriceChange"])) for d in training]
testing_data = [(FreqDist(d["tokens"]), score(d["2-2dayPriceChange"])) for d in testing]

## iniate training features and training labels
training_features = [] ## list of tuples
training_labels = [] ## list of integers

## initate lemmeatizer in order to better cross check words with lexicon values
lemmatizer = WordNetLemmatizer()

## create vector for each TRAINING document
## feature = (valence score,arousal score,dominance score)
## label = 1(down),2(flat),or 3(up)
for d in training_data:
    vTotal=0;aTotal=0;dTotal=0
    tokens = list(d[0].keys())
    for word in tokens:
        word = lemmatizer.lemmatize(word)
        if word in data:
            vTotal += data[word][0]
            aTotal += data[word][1]
            dTotal += data[word][2]
    training_features.append((vTotal,aTotal,dTotal))
    labeltext = d[1]
    if labeltext == "down":
        training_labels.append(1)
    elif labeltext == "flat":
        training_labels.append(2)
    elif labeltext == "up":
        training_labels.append(3)

## repreat process for each TESTING document
testing_features = [] 
testing_labels = []

for d in testing_data:
    vTotal=0;aTotal=0;dTotal=0
    tokens = list(d[0].keys())
    for word in tokens:
        word = lemmatizer.lemmatize(word)
        if word in data:
            vTotal += data[word][0]
            aTotal += data[word][1]
            dTotal += data[word][2]
    testing_features.append((vTotal,aTotal,dTotal))
    labeltext = d[1]
    if labeltext == "down":
        testing_labels.append(1)
    elif labeltext == "flat":
        testing_labels.append(2)
    elif labeltext == "up":
        testing_labels.append(3)

## build and train models

dtc_model = DecisionTreeClassifier()
dtc_model.fit(training_features,training_labels)
dtc_predicted = dtc_model.predict(testing_features)

knc_model = KNeighborsClassifier()
knc_model.fit(training_features,training_labels)
knc_predicted = knc_model.predict(testing_features)

mlpc_model = MLPClassifier()
mlpc_model.fit(training_features,training_labels)
mlpc_predicted = mlpc_model.predict(testing_features)

## print classification reports

print("Accuracy using DecisionTreeClassifier:", metrics.accuracy_score(testing_labels, dtc_predicted))
print(confusion_matrix(testing_labels,dtc_predicted))

print("Accuracy using KNeighborsClassifier:", metrics.accuracy_score(testing_labels, knc_predicted))
print(confusion_matrix(testing_labels,knc_predicted))

print("Accuracy using MLPClassifier:", metrics.accuracy_score(testing_labels, mlpc_predicted))
print(confusion_matrix(testing_labels,mlpc_predicted))



