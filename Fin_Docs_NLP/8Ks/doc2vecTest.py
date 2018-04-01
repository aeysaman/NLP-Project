'''
Created on Apr 1, 2018

@author: aldoj
'''
import pickle

from gensim.models.doc2vec import Doc2Vec, TaggedDocument


with open("8Ks_&_prices.p", "rb") as f:
        docs = pickle.load(f)

taggedDocs = []
for i, x in enumerate(docs):
    taggedDocs += [TaggedDocument(x, [i])]
    
model = Doc2Vec(taggedDocs, vector_size = 100, window = 5, min_count = 5, workers = 4)

doc1 = model.docvecs[0]

print("done")