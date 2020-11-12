f = open("dataset_new.csv")

# -----------------------------------------------------
# calculate word frequency
x = []
for i in range(0,30):
    x.append(0)

y = []
for i in range(0,200):
    y.append(0)

for lines in f:
    cnt = len(lines.split(',')[1].split(' '))
    x[cnt] += 1 

    txt = lines.split(',')[1]
    cntchar = 0

    for t in txt:
        cntchar += 1

    y[cntchar] += 1 
    


print(x)

#calculate average length
avg_len = 0
total = 0
for i in range(0,30):
    avg_len += x[i]*i
    total += x[i]


print("average word length:"+str(round(avg_len/total)))


avg_len = 0
total = 0
for i in range(0,200):
    avg_len += y[i]*i
    total += y[i]

print("average char length:"+str(round(avg_len/total)))

#-----------------------------------------------------

#-----------------------------------------------------

wordlist = {}
f = open("dataset_new.csv")

for lines in f:
    wordstring = lines.split(',')[1].split()
    for words in wordstring:
        if words in wordlist:
            wordlist[words] += 1
        else:
            wordlist[words] = 0

# print(wordlist)

wordfreq = 0
for w in wordlist:

    if wordlist[w] >= 5:
        wordfreq += 1

print(wordfreq)
#-----------------------------------------------------

