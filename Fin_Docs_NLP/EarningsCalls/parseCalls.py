'''
Created on Mar 23, 2018

@author: aldoj
'''
import re

import PyPDF2


loc = "C:/Users/aldoj/git/NLP-Project/Fin_Docs_NLP/EarningsCalls/Raw Transcripts/FS000000002333256681/FS000000002333420627.pdf"
quarterformat = re.compile("Q[1-4] 201[0-9] Earnings Call")
 
obj = open(loc, 'rb')
 
reader = PyPDF2.PdfFileReader(obj)

print(reader.getDocumentInfo())

def asLines(r):
    pages = [page.extractText() for page in r.pages]
    text = "".join(pages)
    return text.splitlines()

def getLine(it, pattern):
    x = None
    while x is None:
        x = re.match(pattern,next(it))
    return x[0]

def linesTill(it, pattern):
    result = []
    x = None
    while x is None:
        line = next(it)
        result +=[line]
        x = re.match(pattern,line)
    return result[:-1]

lines = asLines(reader)
lines = [l for l in lines if re.match("Page \d", l) is None]
foo = iter(lines)

q = getLine(foo, "Q[1-4] 201[0-9] Earnings Call")
getLine(foo, "Company Participants")
company = linesTill(foo, "Other Participants")
other = linesTill(foo, "MANAGEMENT DISCUSSION SECTION")
disc = linesTill(foo, "Q&A")
qa = linesTill(foo, "This transcript may not be 100 percent accurate.*")

print(q)
print(company)
print(other)
print(disc[:10])
print(qa[:10])
print()

is_name = lambda x: x if x in company or x in other or x == "Operator" else None

def classifyDiscussion(text, cur_name = None):
    while cur_name is None:
        cur_name = is_name(next(text))
    
    result = []
    while True:
        try:
            x = next(text)
        except StopIteration:
            return [(cur_name, result)]
        else:
            if is_name(x) is None:
                result += [x]
            else:
                return [(cur_name, result)] + classifyDiscussion(text, x)
    
disc_names = classifyDiscussion(iter(disc))

for name, bar in disc_names:
    print(name, bar)
    
print()

def splitQA(x):
    foo = re.match("<[AQ] - .*>:.*", x)
    if x == "Operator" or x in company or x in other:
        return x, []
    elif not foo is None:
        return x.split(">")[0][5:], [x.split(":")[1]]
    else:
        return None, []
    
    
is_nameQA = lambda x : x if not splitQA(x)[0] is None else None
    
def classifyQA(text, prev = None):
    while prev is None:
        prev = is_nameQA(next(text))

    name, result = splitQA(prev)
    while True:
        try:
            x = next(text)
        except StopIteration:
            return [(name, result)]
        else:
            if is_nameQA(x) is None:
                result +=[x]
            else:
                return [(name, result)] + classifyQA(text, x)

qa_names = classifyQA(iter(qa))

for name, bar in qa_names:
    print(name, bar)