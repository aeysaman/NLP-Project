'''
Created on Mar 30, 2018

@author: aldoj
'''
import csv
from datetime import datetime, timedelta
import glob
import pickle
import re

import nltk
from nltk.corpus import stopwords

stopwords = stopwords.words("english")
stopwords.extend([",", "(", ")", ":", ".", ",", "``", "''", ";", "$"])
stopwords.extend(["on", "'s"])

sector = "Information Technology"
mktcapmax = 1000*10**9
mktcapmin = 10*10**9

folder = "C:/Users/aldoj/Documents/Natural Language Processing/Final Project/"
priceLoc = folder + "All Prices/*"
docLoc = folder + "all 8Ks/*"
constituentLoc = folder + "SPX constituents 2010.csv"
outpotLoc = folder + "8Ks_Tech_10B.p"

test = lambda x : x["GICS Sector"] == sector and x["Market Cap:2010C"] != "--" and int(x["Market Cap:2010C"]) <mktcapmax and int(x["Market Cap:2010C"]) > mktcapmin

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
        x.update({"name":file.split('\\')[-1]})
        del x["raw_text"]
        
    return data

def readAllFiles(names):
    result = []
    for file in glob.glob(docLoc):
        foo = re.split(r"\\", file)[-1]
        if(foo in names):
            print(foo)
            result.extend(readFile(file))
    return result

def readPrices(names):
    data = {}
    for x in glob.glob(priceLoc):
        if(re.split(r"\\", x)[-1][:-4] in names):
            name = re.search(r"\\.*\.",x).group(0).replace(".", "").replace("\\", "")
            with open(x) as f:
                foo = {datetime.strptime(row["Date"], "%Y-%m-%d"): row["Adj Close"] for row in csv.DictReader(f)}
            data.update({name: foo})
    return data
            
def findPrice(name, start, dist, increment):
    x = start
    x += timedelta(days=dist)
    while(not x in prices[name]):
        if(x>datemax[name] or x<datemin[name]):
            return None
        x+= timedelta(days = increment)
    return float(prices[name][x])

def priceChange(name, date, start, end):
    before = findPrice(name, date, start, -1)
    after = findPrice(name, date, end, 1)
    return None if (before is None or after is None) else (after / before) - 1.0;
        
def readNames():
    with open(constituentLoc) as csvfile:
        names = [row["Ticker"] for row in csv.DictReader(csvfile) if test(row)] 

    names = [x.split(" ")[0] for x in names]
    print(len(names), names)
    return names
 

print("start")

names = readNames()

docs = readAllFiles(names)

prices = readPrices(names)

print("done reading")

datemax = {name: max(values.keys()) for name, values in prices.items()}
datemin = {name: min(values.keys()) for name, values in prices.items()}

for x in docs:
    x.update({"date": datetime.strptime(x["time"][:8], "%Y%m%d")})
    x.update({"-2+2dayPriceChange": priceChange(x["name"], x["date"], -2, 2)})
    x.update({"-1+1dayPriceChange": priceChange(x["name"], x["date"], -1, 1)})
    x.update({"+2+10dayPriceChange": priceChange(x["name"], x["date"], 2, 10)})
    
print ("done processing")

with open(outpotLoc, "wb") as f:
    pickle.dump(docs, f)
    
print("done")