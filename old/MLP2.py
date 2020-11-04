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
import keras

from keras.models import *
from keras.models import Model
from keras.preprocessing import text

import warnings
from sklearn.metrics import classification_report,accuracy_score


def train_save_model():

    dic = 3
    maxlen = 20
    summarylen = 100

    #------------------------------------------------------------------------------------------------------------------------------
    # calculate the length of the files..

    #subtract 1 if headers are present..
    num_train = len(open('data/train.csv', 'r').readlines())
    num_valid = len(open('data/valid.csv', 'r').readlines())
    num_test = len(open('data/test.csv', 'r').readlines())

    print('\nDataset statistics : ' + '  num_train : ' + str(num_train) + ',  num_valid  : ' + str(num_valid) + ',  num_test  : ' + str(num_test) + '\n')

    #-------------------------------------------------------------------------------------------------------
    #Loading features..

    colnames  =  ['title','summary','bias']
    df_train = pandas.read_csv('data/train.csv', names=colnames)
    df_valid = pandas.read_csv('data/valid.csv', names=colnames)
    df_test = pandas.read_csv('data/test.csv', names=colnames)

    # train_title_feature = df_train.title.tolist()
    train_summary_feature = df_train.summary.tolist()

    # valid_title_feature = df_valid.title.tolist()
    valid_summary_feature = df_valid.summary.tolist()

    # test_title_feature = df_test.title.tolist()
    test_summary_feature = df_test.summary.tolist()

    title_list_all = train_summary_feature + valid_summary_feature + test_summary_feature

    print('Turning titles into vectors...')
    tokenizer_q = text.Tokenizer(num_words=100)
    tokenizer_q.fit_on_texts(title_list_all)
    wrdidx = tokenizer_q.word_index

    #-------------------------------------------------------------------------------------------------------

    dic_emb1 = {}
    
    file1 = open('data/glove.6B.300d.txt')
    for line in file1:
        values = line.split() # Word and weights separated by space
        word = values[0] # Word is first symbol on each line
        word_weights = np.asarray(values[1:], dtype=np.float32)
        dic_emb1[word] = word_weights

    word2idx = {}
    idx2word = {}

    len_wrdidx = len(wrdidx)+1

    weight_matrix1 = np.zeros((len_wrdidx,300))

    embwrd=[]

    with open('data/glove.6B.300d.txt', 'r') as file: 
        for line in file:
            values = line.split() # Word and weights separated by space
            word = values[0] # Word is first symbol on each line
            if word in wrdidx:
                index = wrdidx.get(word) 
                embwrd.append(word)
                word_weights1 = np.asarray(values[1:], dtype=np.float32)
                word2idx[word] = index
                idx2word[index]=word
                weight_matrix1[index]=word_weights1
                
    file.close()

    for i in wrdidx:
        if i not in embwrd:
            weight_matrix1[wrdidx.get(i)]=np.asarray([0.0]* 300, dtype=np.float32)

    weight_matrixx1 = np.asarray(weight_matrix1)

    #-------------------------------------------------------------------------------------------------------
    # model building..

    print('\nBuilding model...\n')

    encode_summary = Input(shape=(summarylen,))
    embed_summary = Embedding(len_wrdidx,300,input_length=summarylen,weights=[weight_matrixx1],trainable=False)(encode_summary)

    gate_model = Flatten()(embed_summary)
    gate_model = Dropout(0.5)(gate_model)
    
    gate_model = Dense(128, activation='relu')(gate_model) #selu #elu=41
    gate_model = Dropout(0.5)(gate_model)

    gate_model = Dense(3, activation='softmax')(gate_model)

    gate_model = Model(inputs=[encode_summary], outputs=gate_model)
    gate_model.summary()

    #Compile model..
    #Compile model..
    gate_model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=[metrics.categorical_accuracy])

    #save model..
    filepath = 'MLP2/MODEL.hdf5'
    checkpoint = ModelCheckpoint(filepath,verbose=1, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='min')
    callbacks_list = [checkpoint, early_stopping]

    testsum_feature = tokenizer_q.texts_to_sequences(test_summary_feature)
    testsum_feature = sequence.pad_sequences(testsum_feature, summarylen, padding='post', value=0, truncating='post')

    
    if os.path.isfile('MLP2/MODEL.h5') == False:

        trainsum_feature = tokenizer_q.texts_to_sequences(train_summary_feature)
        validsum_feature = tokenizer_q.texts_to_sequences(valid_summary_feature)

        trainsum_feature = sequence.pad_sequences(trainsum_feature, summarylen, padding='post', value=0, truncating='post')
        validsum_feature = sequence.pad_sequences(validsum_feature, summarylen, padding='post', value=0, truncating='post')
        
        train_bias = df_train.bias.tolist()
        train_bias_list = []

        for item in train_bias:
            if item == 'left':
                train_bias_list.append(0)
            elif item == 'right':
                train_bias_list.append(2)
            else:
                train_bias_list.append(1)

        valid_bias = df_valid.bias.tolist()
        valid_bias_list = []

        for item in valid_bias:
            if item == 'left':
                valid_bias_list.append(0)
            elif item == 'right':
                valid_bias_list.append(2)
            else:
                valid_bias_list.append(1)

        trainans = keras.utils.to_categorical(train_bias_list, 3)
        validans = keras.utils.to_categorical(valid_bias_list, 3)

        history = gate_model.fit([trainsum_feature], trainans, epochs=500, batch_size=128, validation_data=([validsum_feature], validans), callbacks=callbacks_list, verbose=1)

        # serialize model to JSON
        model_json = gate_model.to_json()
        with open('MLP2/MODEL.json', "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        gate_model.save_weights('MLP2/MODEL.h5')
        print("\nSaved model to disk...\n")
    else: 
        print('\nLoading model...')  
        # load json and create model
        json_file = open('MLP2/MODEL.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        gate_model = model_from_json(loaded_model_json)
        # load weights into new model
        gate_model.load_weights('MLP2/MODEL.h5', by_name=True) 

    print('\n\nGenerating answers...') 
    ans = gate_model.predict([testsum_feature])

    fp = open('MLP2/test.ans', 'w')

    for h in range(num_test):
        if np.argmax(ans[h]) == 0:
            fp.write('left\n')
        elif np.argmax(ans[h]) == 2:
            fp.write('right\n')
        else:
            fp.write('center\n')

    fp.close()

def evaluate():

    warnings.filterwarnings("ignore", category=UserWarning)
    
    f_test = open('data/test.csv')

    lines_test = f_test.readlines()

    true_ans_test = []

    for line in lines_test:
        bias = line.split(',')[2].strip()
        true_ans_test.append(bias)

    f = open('MLP2/test.ans')

    lines = f.readlines()

    pred_ans = []

    for line in lines:
        pred_ans.append(line.strip())

    f.close()
    print(classification_report(true_ans_test, pred_ans))

    print('\n\n')

def main():

    if os.path.exists('MLP2') == False:
        os.mkdir('MLP2')

    train_save_model()
    evaluate()

    # evaluate()

if __name__ == "__main__":
    main()
