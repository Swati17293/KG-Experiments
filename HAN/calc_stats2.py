f = open("dataset_new.csv")

# -----------------------------------------------------
# calculate word frequency
MAX_WORD_LENGTH = 0
MAX_WORDS = 0
MAX_NB_CHARS = 0

for lines in f:
    cnt = len(lines.split(',')[1].split(' '))
    if cnt > MAX_WORDS:
        MAX_WORDS = cnt 

    txt = lines.split(',')[1]
    cntchar = 0

    for t in txt:
        cntchar += 1

    if cntchar > MAX_NB_CHARS:
        MAX_NB_CHARS = cntchar 

    wrds = txt.split(' ')
    for w in wrds:
        cntchar = 0

        for t in w:
            cntchar += 1

        if cntchar > MAX_WORD_LENGTH:
            MAX_WORD_LENGTH = cntchar


print(MAX_WORD_LENGTH)
print(MAX_WORDS)
print(MAX_NB_CHARS)
#-----------------------------------------------------

