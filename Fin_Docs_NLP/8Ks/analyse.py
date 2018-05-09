'''
Final Project
Authors: Aldo Eysaman, Dominic Medina, and Jose Lorenzo Guevara
Class: Natural Language Processing by Professor Prud'hommeaux
Date: 5/10/2018
'''

from _functools import partial
import glob
import math
import pickle
import warnings

from nltk import FreqDist
from numpy import median, mean
import numpy
from sklearn import metrics
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.svm.classes import LinearSVC
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier

warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
period = "-2+2dayPriceChange"
folder = "C:/Users/aldoj/Documents/Natural Language Processing/Final Project/"

#----------read Data from prepped file------------
def getPrepped(name):
    with open(name, "rb") as f:
        return pickle.load(f)

def getPrepFiles(names):
    print("Files: %s"% " ".join(names))
    return [doc for f in names for doc in getPrepped(folder+f)]

def getAllPrepFiles(loc):
    names = glob.glob(loc)
    print("Files: %s"% " ".join(names))
    return [doc for f in names for doc in getPrepped(f)]

# docs = getAllPrepFiles(folder + '8Ks_ConsDisc*_p2.p')
docs = getPrepFiles(["8Ks_ConsDisc_0-3B_p2.p"])


#-------create training & test data---------
docs = [d for d in docs if not d[period] is None]
docs = [d for d in docs if len(d["tokens"])>1]
print(len(docs))
    
years = [d["date"].year for d in docs]

year = 2008
training = [d for d in docs if d["date"].year <=year][:10]
testing = [d for d in docs if d["date"].year >=year][:10]

print("Training: %d\tTesting: %d" %(len(training), len(testing)))

def score(change):
    if change <-.02:
        return "down"
    elif change <.02:
        return "flat"
    else:
        return "up"

#Generate Results 

train_results = [score(d[period]) for d in training]
test_results = [score(d[period]) for d in testing]
print("done with results")
 
gettopwords = lambda foo, n: [word for word, freq in FreqDist([x for d in foo for x in d["tokens"]]).most_common(n)]

def freqs(n):
    topwords = gettopwords(training, n)
      
    def topCounts(toks):
        freq = FreqDist(toks)
        return [freq[word] if word in freq else 0 for word in topwords]
     
    train_freq = [topCounts(doc["tokens"]) for doc in training]
    test_freq = [topCounts(doc["tokens"]) for doc in testing]
    return train_freq, test_freq
 
def tfidf(n):
    topwords = gettopwords(training, n)
    
    idf = {word: math.log(float(len(training)) / sum([word in d["tokens"] for d in training])) for word in topwords }
     
    def topTFIDF(toks):
        freq = FreqDist(toks)
        return [float(freq[word])*idf[word] if word in freq else 0 for word in topwords]
     
    train_tfidf = [topTFIDF(doc["tokens"]) for doc in training ]
    test_tfidf = [topTFIDF(doc["tokens"]) for doc in testing ]
    return train_tfidf, test_tfidf

def w2vFeatures():
    with open(folder+ "word2vecDict.p", "rb") as f:
            wvDict = pickle.load(f)
     
    def sumvecs(tokens):
        vectors = numpy.matrix([wvDict[w] for w in tokens if w in wvDict])
        return numpy.sum(vectors, axis = 0).flatten().tolist()[0]
     
    train = [sumvecs(doc["tokens"]) for doc in training]
    test= [sumvecs(doc["tokens"]) for doc in testing]
    return train, test

def d2vFeatures():
    with open(folder +"Doc2VecModel.bin", "rb") as f:
        d2vModel = pickle.load(f)
    
    train_doc2vec = [d2vModel.infer_vector(doc["tokens"]) for doc in training]
    test_doc2vec = [d2vModel.infer_vector(doc["tokens"]) for doc in testing]
    
    return train_doc2vec, test_doc2vec

def affective():
    data = {} ## string(word):tuple(valence,arousal,dominance)
    with open("lexicon.csv") as f:
        next(f) ## skips title row of csv file
        for line in f:
            strArray = line.strip().split(",")
            data[strArray[0]] = [float(x) for x in strArray[1:4]]
    
    affectscores = lambda x: [data[w] for w in x["tokens"] if w in data]
     
    train_affect = [affectscores(x) for x in training]
    test_affect = [affectscores(x) for x in testing]
       
    fns = (sum, max, min, median, mean)
     
    affectfeatures = lambda ls: [ [f([foo[i] for foo in x ]) for f in fns for i in range(3)] if len(x) >0 else [0]*3*len(fns) for x in ls]
    train_feat = affectfeatures(train_affect)
    test_feat = affectfeatures(test_affect)
    return train_feat, test_feat
 

#----------------Print Results------------------

amounts = [ (direction, len([ t for t in test_results if t ==direction])) for direction in ["down", "flat", "up"]]
print(amounts)
print("Majority Baseline: %.2f%%" % (float(max([b for a,b in amounts])) / len(test_results)*100))
   
MLP = ("Multi-Layer Perceptron", MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1))
SVM = ("LinearSVC", LinearSVC())
GNB = ("Gaussian Naive Bayes", GaussianNB())
RF = ("Random Forest", RandomForestClassifier())
KNN = ("K Nearest Neighbors", KNeighborsClassifier(5))
ADA = ("Ada Boost", AdaBoostClassifier())
models = [MLP, SVM, GNB, RF, KNN, ADA]
   
   
def runandcompare(ms, train_v, test_v,):
    for name, m in ms:
        m.fit(train_v, train_results)
        pred = m.predict(test_v)
        print("%s: %.2f%%" %(name, 100*metrics.accuracy_score(test_results, pred)))
#         print(metrics.classification_report(test_results, pred))
   
D2V = ("Doc2Vec", d2vFeatures)
W2V = ("Word2Vec", w2vFeatures)
Frequency = ("Frequencies: 1000", partial(freqs, 1000))
TFIDF = ("TF-IDF: 1000", partial(tfidf, 1000))
Affect = ("Affective", affective)

for feat_name, features in [D2V, W2V, Frequency, TFIDF, Affect]:
    print("\n\t" + feat_name)
    train_feat, test_feat = features()
    runandcompare(models, train_feat, test_feat)
