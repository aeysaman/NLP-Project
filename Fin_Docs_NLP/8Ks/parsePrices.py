'''
Created on Mar 30, 2018

@author: aldoj
'''
import csv
from datetime import datetime, timedelta
import glob
import pickle
import re


def readPrices(loc):
    data = {}
    for x in glob.glob(loc):
        name = re.search(r"\\.*\.",x).group(0).replace(".", "").replace("\\", "")
        with open(x) as f:
            foo = {datetime.strptime(row["Date"], "%Y-%m-%d"): row["Adj Close"] for row in csv.DictReader(f)}
        data.update({name: foo})
    return data

def loadDocs(loc):
    with open(loc, "rb") as f:
        docs = pickle.load(f)
     
    for x in docs:
        x.update({"date": datetime.strptime(x["time"][:8], "%Y%m%d")})
        
    return docs

print("start")
prices = readPrices("prices/*")
docs = loadDocs("8Ks_prepped.p")

datemax = {name: max(values.keys()) for name, values in prices.items()}
datemin = {name: min(values.keys()) for name, values in prices.items()}

def findPrice(name, start, dist, increment):
    x = start
    x += timedelta(days=dist)
    while(not x in prices[name]):
        if(x>datemax[name] or x<datemin[name]):
            return None
        x+= timedelta(days = increment)
    return float(prices[name][x])

def priceChange(name, date, daysbefore, daysafter):
    before = findPrice(name, date, -daysbefore, -1)
    after = findPrice(name, date, daysafter, 1)
    return None if (before is None or after is None) else (after / before) - 1.0;

for x in docs:
    x.update({"2-2dayPriceChange": priceChange(x["name"], x["date"], 2, 2)})
    x.update({"2-5dayPriceChange": priceChange(x["name"], x["date"], 2, 5)})
    
with open("8Ks_&_prices.p", "wb") as f:
    pickle.dump(docs, f)
    
print("done")