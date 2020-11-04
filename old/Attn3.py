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

    train_title_feature = df_train.title.tolist()
    train_summary_feature = df_train.summary.tolist()

    valid_title_feature = df_valid.title.tolist()
    valid_summary_feature = df_valid.summary.tolist()

    test_title_feature = df_test.title.tolist()
    test_summary_feature = df_test.summary.tolist()

    title_list_all = train_title_feature + valid_title_feature + test_title_feature
    summary_list_all = train_summary_feature + valid_summary_feature + test_summary_feature

    print('Turning titles into vectors...')
    tokenizer_q = text.Tokenizer(num_words=1000)
    tokenizer_q.fit_on_texts(title_list_all)
    wrdidx_title = tokenizer_q.word_index

    tokenizer_s = text.Tokenizer(num_words=700)
    tokenizer_s.fit_on_texts(summary_list_all)
    wrdidx_summary = tokenizer_s.word_index

    #-------------------------------------------------------------------------------------------------------

    dic_emb1 = {}
    
    file1 = open('data/glove.6B.300d.txt')
    for line in file1:
        values = line.split() # Word and weights separated by space
        word = values[0] # Word is first symbol on each line
        word_weights = np.asarray(values[1:], dtype=np.float32)
        dic_emb1[word] = word_weights

    #-------------------------------------------------------------------------------------------------------

    len_wrdidx_title = len(wrdidx_title)+1

    weight_matrix1 = np.zeros((len_wrdidx_title,300))

    embwrd=[]

    with open('data/glove.6B.300d.txt', 'r') as file: 
        for line in file:
            values = line.split() # Word and weights separated by space
            word = values[0] # Word is first symbol on each line
            if word in wrdidx_title:
                index = wrdidx_title.get(word) 
                embwrd.append(word)
                word_weights1 = np.asarray(values[1:], dtype=np.float32)
                weight_matrix1[index]=word_weights1
                
    file.close()

    for i in wrdidx_title:
        if i not in embwrd:
            weight_matrix1[wrdidx_title.get(i)]=np.asarray([0.0]* 300, dtype=np.float32)

    weight_matrixx1 = np.asarray(weight_matrix1)

    #-------------------------------------------------------------------------------------------------------

    len_wrdidx_summary = len(wrdidx_summary)+1

    weight_matrix2 = np.zeros((len_wrdidx_summary,300))

    embwrd=[]

    with open('data/glove.6B.300d.txt', 'r') as file: 
        for line in file:
            values = line.split() # Word and weights separated by space
            word = values[0] # Word is first symbol on each line
            if word in wrdidx_summary:
                index = wrdidx_summary.get(word) 
                embwrd.append(word)
                word_weights1 = np.asarray(values[1:], dtype=np.float32)
                weight_matrix2[index]=word_weights1
                
    file.close()

    for i in wrdidx_summary:
        if i not in embwrd:
            weight_matrix2[wrdidx_summary.get(i)]=np.asarray([0.0]* 300, dtype=np.float32)

    weight_matrixx2 = np.asarray(weight_matrix2)

    #-------------------------------------------------------------------------------------------------------

    # model building..

    print('\nBuilding model...\n')

    encode_title = Input(shape=(maxlen,))
    encode_summary = Input(shape=(summarylen,))

    embed_title = Embedding(len_wrdidx_title,300,input_length=maxlen,weights=[weight_matrixx1],trainable=False)(encode_title)
    embed_summary = Embedding(len_wrdidx_summary,300,input_length=summarylen,weights=[weight_matrixx2],trainable=False)(encode_summary)

    embed_title = Dense(128, activation='relu')(embed_title) #selu #elu=41
    embed_title = Dropout(0.5)(embed_title)

    embed_title = Flatten()(embed_title)
    embed_title = Dropout(0.5)(embed_title)

    #-------------------------------------------------------------------------------------------------------
    # Recurrent Layers (2)
    # try playing around with the hidden size of the recurrent layers, the batch size in training process, or the  param @window_width if using a 'local' attention.

    config = 1

    # Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.2))(embed_title)
    
    if config != 0:
        encoder_output, hidden_state = GRU(units=128, return_sequences=True, return_state=True, dropout=0.5, recurrent_dropout=0.2)(embed_summary)
        attention_input = [encoder_output, hidden_state]

    # Optional Attention Mechanisms
    if config == 1:
        encoder_output, attention_weights = SelfAttention(size=128, num_hops=10, use_penalization=False)(encoder_output)
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

    merge_model = Concatenate()([embed_title, encoder_output])
    merge_model = Dense(128, activation='tanh')(merge_model)
    merge_model = Dropout(0.5)(merge_model)

    gate_model = Dense(3, activation='softmax')(merge_model)

    gate_model = Model(inputs=[encode_title, encode_summary], outputs=gate_model)
    gate_model.summary()

    #Compile model..
    #Compile model..
    gate_model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=[metrics.categorical_accuracy])

    #save model..
    filepath = 'Attn3/MODEL.hdf5'
    checkpoint = ModelCheckpoint(filepath,verbose=1, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=8, mode='min')
    callbacks_list = [checkpoint, early_stopping]

    testque_feature = tokenizer_q.texts_to_sequences(test_title_feature)
    testque_feature = sequence.pad_sequences(testque_feature, maxlen, padding='post', value=0, truncating='post')

    testsum_feature = tokenizer_q.texts_to_sequences(test_summary_feature)
    testsum_feature = sequence.pad_sequences(testsum_feature, summarylen, padding='post', value=0, truncating='post')

    
    if os.path.isfile('Attn3/MODEL.h5') == False:

        trainque_feature = tokenizer_q.texts_to_sequences(train_title_feature)
        validque_feature = tokenizer_q.texts_to_sequences(valid_title_feature)

        trainsum_feature = tokenizer_q.texts_to_sequences(train_summary_feature)
        validsum_feature = tokenizer_q.texts_to_sequences(valid_summary_feature)

        trainque_feature = sequence.pad_sequences(trainque_feature, maxlen, padding='post', value=0, truncating='post')
        validque_feature = sequence.pad_sequences(validque_feature, maxlen, padding='post', value=0, truncating='post')

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

        trainans = to_categorical(train_bias_list, 3)
        validans = to_categorical(valid_bias_list, 3)

        history = gate_model.fit([trainque_feature, trainsum_feature], trainans, epochs=500, batch_size=128, validation_data=([validque_feature, validsum_feature], validans), callbacks=callbacks_list, verbose=1)

        # serialize model to JSON
        model_json = gate_model.to_json()
        with open('Attn3/MODEL.json', "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        gate_model.save_weights('Attn3/MODEL.h5')
        print("\nSaved model to disk...\n")
    else: 
        print('\nLoading model...')  
        # load json and create model
        json_file = open('Attn3/MODEL.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        gate_model = model_from_json(loaded_model_json)
        # load weights into new model
        gate_model.load_weights('Attn3/MODEL.h5', by_name=True) 

    print('\n\nGenerating answers...') 
    ans = gate_model.predict([testque_feature, testsum_feature])

    fp = open('Attn3/test.ans', 'w')

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

    f = open('Attn3/test.ans')

    lines = f.readlines()

    pred_ans = []

    for line in lines:
        pred_ans.append(line.strip())

    f.close()
    print(classification_report(true_ans_test, pred_ans))

    print('\n\n')

def main():

    if os.path.exists('Attn3') == False:
        os.mkdir('Attn3')

    train_save_model()
    evaluate()

    # evaluate()

if __name__ == "__main__":
    main()
