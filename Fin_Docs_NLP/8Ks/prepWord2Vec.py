'''
Final Project
Authors: Aldo Eysaman, Dominic Medina, and Jose Lorenzo Guevara
Class: Natural Language Processing by Professor Prud'hommeaux
Date: 5/10/2018
'''
import pickle

import gensim


#read Data from prepped file
fnames = ["8Ks_ConsDisc_10-1000B_p2.p", "8Ks_ConsDisc_5-10B_p2.p", 
          "8Ks_ConsDisc_3-5B_p2.p", "8Ks_ConsDisc_0-3B_p2.p",
          "8Ks_Tech_0-3B_p2.p", "8Ks_Tech_3-5B_p2.p", 
          "8Ks_Tech_5-10B_p2.p", "8Ks_Tech_10-1000B_p2.p",
          "8Ks_ConsStpl_0-10B_p2.p", "8Ks_ConsStpl_10-1000B_p2.p"]
print("File: %s"% ", ".join(fnames))

def getWords():
    for name in fnames:
        with open("C:/Users/aldoj/Documents/Natural Language Processing/Final Project/"+name, "rb") as f:
            for doc in pickle.load(f):
                for word in doc["tokens"]:
                    yield word
    
model = gensim.models.KeyedVectors.load_word2vec_format("C:/Users/aldoj/Documents/Natural Language Processing/Final Project/GoogleNews-vectors-negative300-SLIM.bin", binary = True)
print("done loading model")

uniquewords = set(getWords())

values = {word:model[word] for word in uniquewords if word in model}

with open("C:/Users/aldoj/Documents/Natural Language Processing/Final Project/word2vecDict.p", "wb") as f:
    pickle.dump(values, f, protocol=2)
    
print("finished")