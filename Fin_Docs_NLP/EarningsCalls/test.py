'''
Created on Mar 23, 2018

@author: aldoj
'''
import PyPDF2
import re
 
loc = "C:/Users/aldoj/eclipse-workspace/EarningsCalls/Raw Transcripts/FS000000002333256681/FS000000002333420627.pdf"
quarterformat = re.compile("Q[1-4] 201[0-9] Earnings Call")
 
obj = open(loc, 'rb')
 
reader = PyPDF2.PdfFileReader(obj)

print(reader.getDocumentInfo())

raw = ""
for page in reader.pages:
    raw +=page.extractText()

quarter = "not found"

for line in raw.splitlines():
    print (line)
#     if line
# print (raw)

