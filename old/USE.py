import tensorflow_hub as hub
import os
import csv
import numpy as np
import pandas

from keras.models import *

from keras import metrics, Sequential
from keras.layers import *
from keras import optimizers

from keras.preprocessing import text
from keras.utils import to_categorical
from keras.preprocessing import sequence

from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping

import tensorflow as tf

from keras.models import *
from keras.models import Model
from keras.preprocessing import text

import warnings
from sklearn.metrics import classification_report,accuracy_score


def train_save_model():

    dic = 3

    #------------------------------------------------------------------------------------------------------------------------------
    # calculate the length of the files..

    #subtract 1 if headers are present..
    num_train = len(open('data/raw/Train.tsv', 'r').readlines())
    num_valid = len(open('data/raw/Valid.tsv', 'r').readlines())
    num_test = len(open('data/raw/Test.tsv', 'r').readlines())

    print('\nDataset statistics : ' + '  num_train : ' + str(num_train) + ',  num_valid  : ' + str(num_valid) + ',  num_test  : ' + str(num_test) + '\n')

    #-------------------------------------------------------------------------------------------------------
    #Loading lists..

    train_ans, valid_ans, anslist = [], [], []

    def ans_vec(data):

        f = open('data/raw/' + data + '.tsv')

        lines = f.readlines()

        for line in lines:

            line = line.split('\t')
            bias = line[2]

            anslist.append(bias)
            if data == 'Train':
                train_ans.append(bias)
            elif data == 'Valid':
                valid_ans.append(bias)

        f.close()

    ans_vec('Train')
    ans_vec('Valid')

    #-------------------------------------------------------------------------------------------------------
    #Loading features..

    embed = hub.load("https://tfhub.dev/google/tf2-preview/nnlm-en-dim128-with-normalization/1")

    colnames = ['title','summary','bias']
    df_train = pandas.read_csv('data/raw/Train.tsv', names=colnames, sep='\t')
    df_valid = pandas.read_csv('data/raw/Valid.tsv', names=colnames, sep='\t')
    df_test = pandas.read_csv('data/raw/Test.tsv', names=colnames, sep='\t')

    train_title_feature = embed(df_train.title.tolist())
    train_summary_feature = embed(df_train.summary.tolist())

    valid_title_feature = embed(df_valid.title.tolist())
    valid_summary_feature = embed(df_valid.summary.tolist())

    test_title_feature = embed(df_test.title.tolist())
    test_summary_feature = embed(df_test.summary.tolist())

    tokenizer_a = text.Tokenizer()
    tokenizer_a.fit_on_texts(anslist)

    trainans_feature = tokenizer_a.texts_to_sequences(train_ans)
    trainans_feature = sequence.pad_sequences(trainans_feature, 3, padding='post', value=0, truncating='post')
    trainans_hot = to_categorical(trainans_feature, dic+1)  #one-hot

    validans_feature = tokenizer_a.texts_to_sequences(valid_ans)
    validans_feature = sequence.pad_sequences(validans_feature, 3, padding='post', value=0, truncating='post')
    validans_hot = to_categorical(validans_feature, dic+1)  #one-hot

    print(validans_hot)

    #-------------------------------------------------------------------------------------------------------
    # model building..

    print('\nBuilding model...\n')

    encode_title = Input(shape=(128,))
    encode_summary = Input(shape=(128,))


    

    merged = Concatenate()([encode_title, encode_summary])

    gate_model =Reshape((1,256))(merged)
    gate_model = Dense(64, activation='elu')(gate_model) #selu #elu=41
    
    gate_model = Dense(dic)(gate_model)
    gate_model = Permute((2, 1))(gate_model)

    
    gate_model = Dense(dic+1, activation='softmax')(gate_model)

    gate_model = Model(inputs=[encode_title, encode_summary], outputs=gate_model)
    gate_model.summary()

    #Compile model..
    #Compile model..
    gate_model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=[metrics.categorical_accuracy])

    #save model..
    filepath = 'models/USE/MODEL.hdf5'
    checkpoint = ModelCheckpoint(filepath,verbose=1, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=400, mode='min')
    callbacks_list = [checkpoint, early_stopping]

    
    if os.path.isfile('models/USE/MODEL.h5') == False:

        history = gate_model.fit([ train_title_feature,train_summary_feature], trainans_hot, epochs=500, batch_size=128, validation_data=([valid_title_feature,valid_summary_feature], validans_hot), callbacks=callbacks_list, verbose=1)

        # serialize model to JSON
        model_json = gate_model.to_json()
        with open('models/USE/MODEL.json', "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        gate_model.save_weights('models/USE/MODEL.h5')
        print("\nSaved model to disk...\n")
    else: 
        print('\nLoading model...')  
        # load json and create model
        json_file = open('models/USE/MODEL.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        gate_model = model_from_json(loaded_model_json)
        # load weights into new model
        gate_model.load_weights('models/USE/MODEL.h5', by_name=True) 

    
    print('\n\nGenerating answers...') 

    if os.path.exists('reports') == False:
        os.mkdir('reports')

    if os.path.isfile('reports/TestUSE.ans') == False:

        test_title_feature = np.load('data/vectorized/Test_title.npy')
        test_summary_feature = np.load('data/vectorized/Test_summary.npy')

        dic_a = tokenizer_a.word_index
        ind_a ={value:key for key, value in dic_a.items()}

        num_test = len(open('data/raw/Test.tsv', 'r').readlines())
        
        ans = gate_model.predict([ test_title_feature, test_summary_feature])
        fp = open('reports/TestUSE.ans', 'w')
        for h in range(num_test):
            i = h
            
            if np.argmax(ans[i][0],axis=0) == 0:
                fp.write('center\n')  #Low frequency words are replaced with "center"
            else:
                for j in range(dic):
                    an = np.argmax(ans[i][j],axis=0)
                    if j != dic-1:
                        anext = np.argmax(ans[i][j+1],axis=0)
                        if an != 0 and anext != 0:  #Words before and after
                            if an == anext:
                                fp.write('')  #Delete duplicate words
                            else:
                                fp.write(ind_a[an] + ' ')
                        elif an != 0 and anext == 0:
                            fp.write(ind_a[an])
                        elif an == 0 and anext != 0: 
                            fp.write(ind_a[anext])
                        else:
                            fp.write('')
                    else:
                        if an != 0:
                            fp.write(ind_a[an] + '\n')
                        else:
                            fp.write('\n')
        fp.close()
    print('\nAnswer generation complete...\n\n')

def evaluate():

    warnings.filterwarnings("ignore", category=UserWarning)
    
    f_test = open('data/raw/Test.tsv')

    lines_test = f_test.readlines()

    true_ans_test = []

    for line in lines_test:
        bias = line.split('\t')[2].strip()
        true_ans_test.append(bias)

    f = open('reports/TestUSE.ans')

    lines = f.readlines()

    pred_ans = []

    for line in lines:
        pred_ans.append(line.strip())

    f.close()
    print(classification_report(true_ans_test, pred_ans))

    print('\n\n')

def main():

    if os.path.exists('models/USE') == False:
        os.mkdir('models/USE')

    train_save_model()

    evaluate()

if __name__ == "__main__":
    main()
