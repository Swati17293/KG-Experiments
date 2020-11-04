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

def text_vectorization(data,dataset):

    # module_url = "https://tfhub.dev/google/nnlm-en-dim128/2"
    module_url = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128-with-normalization/1"
    
    embed = hub.KerasLayer(module_url)

    colnames = ['title','summary','bias']
    df = pandas.read_csv('data/'+dataset+'/' + data + '.csv', names=colnames, sep='\t')

    lst_title = df.title.tolist()
    
    #Vectorization of titles..
    embeddings_title = embed(lst_title)
    embeddings_title = tf.make_tensor_proto(embeddings_title)
    embeddings_title = tf.make_ndarray(embeddings_title)

    embeddings_title = np.array(embeddings_title) 
    np.save(data + '_title.npy', embeddings_title) 

    
def train_save_model(dataset,model_name):

    dic = 5

    #------------------------------------------------------------------------------------------------------------------------------
    # calculate the length of the files..

    #subtract 1 if headers are present..
    num_train = len(open('data/'+ dataset +'/train.csv', 'r').readlines())
    num_valid = len(open('data/'+ dataset +'/valid.csv', 'r').readlines())
    num_test = len(open('data/'+ dataset +'/test.csv', 'r').readlines())

    print('\nDataset statistics : ' + '  num_train : ' + str(num_train) + ',  num_valid  : ' + str(num_valid) + ',  num_test  : ' + str(num_test) + '\n')
    #-------------------------------------------------------------------------------------------------------
    # model building..

    print('\nBuilding model...\n')

    #title model..
    encode_title = Input(shape=(128,))
    gate_model = Dense(128, activation='relu')(encode_title) 
    gate_model = Dropout(0.5)(gate_model)
    
    gate_model = Dense(5, activation='softmax')(gate_model)

    gate_model = Model(inputs=[encode_title], outputs=gate_model)
    gate_model.summary()
    
    #Compile model..
    gate_model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=[metrics.categorical_accuracy])

    #save model..
    filepath = 'models/'+ model_name +'/MODEL.hdf5'
    checkpoint = ModelCheckpoint(filepath,verbose=1, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    callbacks_list = [checkpoint, early_stopping]

    
    if os.path.isfile('models/'+ model_name +'/MODEL.h5') == False:

        colnames  =  ['title','summary','bias']
        df_train = pandas.read_csv('data/'+ dataset +'/train.csv', names=colnames, sep='\t')
        df_valid = pandas.read_csv('data/'+ dataset +'/valid.csv', names=colnames, sep='\t')
        df_test = pandas.read_csv('data/'+ dataset +'/test.csv', names=colnames, sep='\t')

        train_bias = df_train.bias.tolist()
        train_bias_list = []

        for item in train_bias:
            if item == 'Lean Left':
                train_bias_list.append(0)
            elif item == 'Left':
                train_bias_list.append(1)
            elif item == 'Center':
                train_bias_list.append(2)
            elif item == 'Lean Right':
                train_bias_list.append(3)
            else:
                train_bias_list.append(4)

        valid_bias = df_valid.bias.tolist()
        valid_bias_list = []

        for item in valid_bias:
            if item == 'Lean Left':
                valid_bias_list.append(0)
            elif item == 'Left':
                valid_bias_list.append(1)
            elif item == 'Center':
                valid_bias_list.append(2)
            elif item == 'Lean Right':
                valid_bias_list.append(3)
            else:
                valid_bias_list.append(4)

        trainans = to_categorical(train_bias_list, 5)
        validans = to_categorical(valid_bias_list, 5)

        trainque_feature = np.load('train_title.npy')
        validque_feature = np.load('valid_title.npy')
        testque_feature = np.load('test_title.npy')

        gate_model.fit([trainque_feature], trainans, epochs=500, batch_size=128, validation_data=([validque_feature], validans), callbacks=callbacks_list, verbose=1)

        # serialize model to JSON
        model_json = gate_model.to_json()
        with open('models/'+ model_name +'/MODEL.json', 'w') as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        gate_model.save_weights('models/'+ model_name +'/MODEL.h5')
        print("\nSaved model to disk...\n")
    else: 
        print('\nLoading model...')  
        # load json and create model
        json_file = open('models/'+ model_name +'/MODEL.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        gate_model = model_from_json(loaded_model_json)
        # load weights into new model
        gate_model.load_weights('models/'+ model_name +'/MODEL.h5', by_name=True) 

    print('\n\nGenerating answers...') 
    ans = gate_model.predict([testque_feature])

    fp = open('models/'+ model_name +'/test.ans', 'w')

    for h in range(num_test):
        if np.argmax(ans[h]) == 0:
            fp.write('Lean Left\n')
        elif np.argmax(ans[h]) == 1:
            fp.write('Left\n')
        elif np.argmax(ans[h]) == 2:
            fp.write('Center\n')
        elif np.argmax(ans[h]) == 3:
            fp.write('Lean Right\n')
        else:
            fp.write('Right\n')

    fp.close()

def evaluate(dataset,model_name):

    warnings.filterwarnings("ignore", category=UserWarning)
    
    f_test = open('data/'+ dataset +'/test.csv')

    lines_test = f_test.readlines()

    true_ans_test = []

    for line in lines_test:
        bias = line.split('\t')[2].strip()
        true_ans_test.append(bias)

    f = open('models/'+ model_name +'/test.ans')

    lines = f.readlines()

    pred_ans = []

    for line in lines:
        pred_ans.append(line.strip())

    f.close()
    print(classification_report(true_ans_test, pred_ans))

    print('\n\n')

def main():

    try:
        print('\n\nTurning text into vectors...')

        if os.path.isfile('test_title.npy') == False:

            text_vectorization('train','KG3')
            text_vectorization('valid','KG3')
            text_vectorization('test','KG3')
        
        print('\nVectorization complete...\n\n')
    except:
        pass


    if os.path.exists('models/USE2/') == False:
        os.mkdir('models/USE2/')
        train_save_model("KG3","USE2")
        evaluate("KG3","USE2")

if __name__ == "__main__":
    main()