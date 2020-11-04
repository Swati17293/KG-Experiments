import tensorflow as tf
import tensorflow_hub as hub

import os, csv, pandas
import numpy as np

from tensorflow.keras import metrics, Sequential
from tensorflow.keras.layers import *

from tensorflow.keras.preprocessing import text, sequence
from keras.utils import to_categorical

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from tensorflow.keras.models import Model, model_from_json

import warnings
from sklearn.metrics import classification_report,accuracy_score

from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers

from layers import Attention, SelfAttention



def train_save_model():

    dic = 3
    maxlen = 20

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

    train_title_feature = df_train.title.tolist()
    # train_summary_feature = df_train.summary.tolist()

    valid_title_feature = df_valid.title.tolist()
    # valid_summary_feature = df_valid.summary.tolist()

    test_title_feature = df_test.title.tolist()
    # test_summary_feature = df_test.summary.tolist()

    title_list_all = train_title_feature + valid_title_feature + test_title_feature

    print('Turning titles into vectors...')
    tokenizer_q = text.Tokenizer(num_words=1000)
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

    encode_title = Input(shape=(maxlen,))
    embedded = Embedding(len_wrdidx,300,input_length=maxlen,weights=[weight_matrixx1],trainable=False)(encode_title)

    #-------------------------------------------------------------------------------------------------------
    # Recurrent Layers (2)
    # try playing around with the hidden size of the recurrent layers, the batch size in training process, or the  param @window_width if using a 'local' attention.

    config = 3
    
    if config != 0:
        encoder_output, hidden_state, cell_state = LSTM(units=128,
                                                            return_sequences=True,
                                                            return_state=True)(embedded)
        attention_input = [encoder_output, hidden_state]
    else:
        encoder_output = LSTM(units=128)(embedded)

    # Optional Attention Mechanisms
    if config == 1:
        encoder_output, attention_weights = SelfAttention(size=128,
                                                        num_hops=10,
                                                        use_penalization=False)(encoder_output)
    elif config == 2:
        encoder_output, attention_weights = Attention(context='many-to-one',
                                                    alignment_type='global')(attention_input)
        encoder_output = Flatten()(encoder_output)
    elif config == 3:
        encoder_output, attention_weights = Attention(context='many-to-one',
                                                    alignment_type='local-p*',
                                                    window_width=100,
                                                    score_function='scaled_dot')(attention_input)
        encoder_output = Flatten()(encoder_output)

    #-------------------------------------------------------------------------------------------------------

    # gate_model = Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.2, return_sequences=False))(embedded)
    gate_model = Dense(3, activation='softmax')(encoder_output)

    gate_model = Model(inputs=[encode_title], outputs=gate_model)
    gate_model.summary()
    
    #Compile model..
    gate_model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=[metrics.categorical_accuracy])

    #save model..
    filepath = 'Attn1/MODEL.hdf5'
    checkpoint = ModelCheckpoint(filepath,verbose=1, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    callbacks_list = [checkpoint, early_stopping]

    testque_feature = tokenizer_q.texts_to_sequences(test_title_feature)
    testque_feature = sequence.pad_sequences(testque_feature, maxlen, padding='post', value=0, truncating='post')

    
    if os.path.isfile('Attn1/MODEL.h5') == False:

        trainque_feature = tokenizer_q.texts_to_sequences(train_title_feature)
        validque_feature = tokenizer_q.texts_to_sequences(valid_title_feature)
        

        trainque_feature = sequence.pad_sequences(trainque_feature, maxlen, padding='post', value=0, truncating='post')
        validque_feature = sequence.pad_sequences(validque_feature, maxlen, padding='post', value=0, truncating='post')
        

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

        trainans = to_categorical(train_bias_list, 3)
        validans = to_categorical(valid_bias_list, 3)

        gate_model.fit([trainque_feature], trainans, epochs=500, batch_size=128, validation_data=([validque_feature], validans), callbacks=callbacks_list, verbose=1)

        # serialize model to JSON
        model_json = gate_model.to_json()
        with open('Attn1/MODEL.json', "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        gate_model.save_weights('Attn1/MODEL.h5')
        print("\nSaved model to disk...\n")
    else: 
        print('\nLoading model...')  
        # load json and create model
        json_file = open('Attn1/MODEL.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        gate_model = model_from_json(loaded_model_json)
        # load weights into new model
        gate_model.load_weights('Attn1/MODEL.h5', by_name=True) 

    print('\n\nGenerating answers...') 
    ans = gate_model.predict([testque_feature])

    fp = open('Attn1/test.ans', 'w')

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

    f = open('Attn1/test.ans')

    lines = f.readlines()

    pred_ans = []

    for line in lines:
        pred_ans.append(line.strip())

    f.close()
    print(classification_report(true_ans_test, pred_ans))

    print('\n\n')

def main():

    if os.path.exists('Attn1/') == False:
        os.mkdir('Attn1/')

    train_save_model()
    evaluate()

    # evaluate()

if __name__ == "__main__":
    main()
