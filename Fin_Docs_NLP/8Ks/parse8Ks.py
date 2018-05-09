'''
Final Project
Authors: Aldo Eysaman, Dominic Medina, and Jose Lorenzo Guevara
Class: Natural Language Processing by Professor Prud'hommeaux
Date: 5/10/2018
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

sector = "Health Care"
mktcapmax = 1000
mktcapmin = 0
mktcapmultiple = 10**9

folder = "C:/Users/aldoj/Documents/Natural Language Processing/Final Project/"
priceLoc = folder + "All Prices/*"
# docLoc = folder + "all 8Ks/*"
docLoc = folder + "all 8Ks/*"
constituentLoc = folder + "SPX constituents 2010.csv"
outpotLoc = folder + "8Ks_ConsDisc_"+ str(mktcapmin) + "-" +str(mktcapmax) +"B_p2 - test.p"

test = lambda x : x["GICS Sector"] == sector and x["Market Cap:2010C"] != "--" and int(x["Market Cap:2010C"]) <mktcapmax*mktcapmultiple and int(x["Market Cap:2010C"]) > mktcapmin*mktcapmultiple

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
        
    for x in data[:10]:
        x.update({"tokens": cleanWords(nltk.word_tokenize(x["raw_text"]))})
        x.update({"name":file.split('\\')[-1]})
        print(x["tokens"])
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
    pickle.dump(docs, f, protocol=2)
    
print("done")