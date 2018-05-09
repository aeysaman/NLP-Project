'''
Final Project
Authors: Aldo Eysaman, Dominic Medina, and Jose Lorenzo Guevara
Class: Natural Language Processing by Professor Prud'hommeaux
Date: 5/10/2018
'''

import pickle
import glob

from gensim.models.doc2vec import TaggedDocument, Doc2Vec

fnames = ['C:/Users/aldoj/Documents/Natural Language Processing/Final Project/8Ks_Tech_0-3B_p2.p']
# fnames = glob.glob('C:/Users/aldoj/Documents/Natural Language Processing/Final Project/8Ks_Tech_*_p2.p')
print("File: %s"% ", ".join(fnames))

def getDocs():
    for name in fnames:
        with open(name, "rb") as f:
            for doc in pickle.load(f):
                yield doc["tokens"]
                
tagged = [TaggedDocument(doc, [i]) for i, doc in enumerate(getDocs())]

print("read docs")
    
model = Doc2Vec(tagged, vector_size = 100, window = 8, min_count=5, workers = 4)

model.save("Doc2VecModel.bin")

print("done")