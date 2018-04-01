'''
Created on Mar 30, 2018

@author: aldoj
'''
import re
from nltk.corpus import stopwords
import nltk
import glob
import pickle

stopwords = stopwords.words("english")
stopwords.extend([",", "(", ")", ":", ".", ",", "``", "''", ";", "$"])
stopwords.extend(["on", "'s"])


def getDocument(f):
    result = []
    x = " "
    while( not x == "<DOCUMENT>\n"):
        x = f.readline()
        if x == "":
            return [], f
    
    while(not x == "</DOCUMENT>\n"):
        x = f.readline()
        result +=[x]
    
    return result, f
   
def readDocuments(file):
    with open(file) as f:
        result = []
        while(True):
            foo, f = getDocument(f)
            if (len(foo) ==0):
                break
            result +=[foo]
        return result
      
def cleanDocument(full):
    result = {}
    text = ""
    for x in full:
        if(not re.match("[A-Z]+:.*", x) is None):
            name, value = x.split(":", 1)
            result.update({name.lower():value.rstrip()})
        else:
            text += x
    
    result.update({"raw_text": re.sub("\s", " ", text).lower()})
    return result

cleanWords = lambda x : [y for y in x if (not y in stopwords) and (re.match("\d", y) is None) and len(y) >1]

def readFile(file):
    data = [cleanDocument(x) for x in readDocuments(file)]
        
    for x in data:
        x.update({"tokens": cleanWords(nltk.word_tokenize(x["raw_text"]))})
        
    return data

alldata = []
for file in glob.glob("./raw/*/*"):
    for x in readFile(file):
        x.update({"name":file.split('\\')[-1]})
        del x["raw_text"]
        alldata.append(x)

print("done")

with open("8Ks_prepped.p", "wb") as f:
    pickle.dump(alldata, f)
    
    