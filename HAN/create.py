#used for lemmatization
from textblob import Word 
import re
import csv

def get_stopwords():
    stops = []
    fstop = open('stopwords.txt','r')
    lines = fstop.readlines()
    for line in lines:
        line = line.strip()
        stops.append(line)
    fstop.close()
    return(stops)

def text_process(text, stops):

    text_lst = text.split(' ')

    wrd_lst = []

    for txt in text_lst: #Stop word deletion
        w = txt
        for div in stops: 
            if txt == div:
                w = ''
        wrd_lst.append(w)

    j = len(wrd_lst)
    for i in range(j):
        if re.match(r'[0-9]+', wrd_lst[i]):  #Digital replacement
            wrd_lst[i] = 'num'
        wrd_lst[i] = Word(wrd_lst[i]).lemmatize()

    wrd_lst_ = []

    for txt in wrd_lst: #Stop word deletion
        w = txt
        for div in stops: 
            if txt == div:
                w = ''
        wrd_lst_.append(w)

    text = ' '.join(wrd_lst_)
    text = text.replace('-',' ')
    text = ''.join([c for c in text if c.isalnum() or c.isspace()])
    
    text = text.lower()
    text = ' '.join(text.split()) #remove extra spaces

    if text == '':
        text = 'empty'

    return(text)


f_new = open('dataset_new.csv','w')

f = open('dataset.csv')

stops = get_stopwords()

csv_reader = csv.reader(f, delimiter=',')

for row in csv_reader:

    title = text_process(row[1], stops)

    line_new = row[0]+','+title+','+row[2]+','+row[3]+','+row[4]+','+row[5]+','+row[6]+'\n'

    f_new.write(line_new)

f.close()
f_new.close()
