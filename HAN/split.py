from sklearn.model_selection import train_test_split

fr = open('dataset_new.csv')
ft = open('train.csv','w')
fs = open('test.csv','w')

ln  = fr.readlines()

A, B = train_test_split(ln, test_size=0.1, random_state=0)

print(len(A))
print(len(B))

for x in B:
    fs.write(x)

for x in A:
    ft.write(x)


    
