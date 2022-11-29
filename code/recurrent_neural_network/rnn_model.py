import numpy as np
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import GRU
from keras.layers import Embedding
from sklearn import metrics
import pandas as pd

from config import configs
from utils.helper import Helper
from utils.vectorizer import Vectorizer

class RNNModel:
    def build_model(self,word_index, embeddings_index, n_classes, EMBEDDING_DIM = 50, dropout = 0.5):
        model = Sequential()

        embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                if len(embedding_matrix[i]) != len(embedding_vector):                    
                    exit(1)
                embedding_matrix[i] = embedding_vector
        
        #Add Layers
        model.add(Embedding(len(word_index) + 1, 
                            EMBEDDING_DIM, 
                            weights = [embedding_matrix], 
                            input_length = configs.MAXIMUM_NUMBER_WORDS, 
                            trainable = True))
        
        for i in range(0,configs.RNN_NUMBER_HIDDEN_LAYERS):
            model.add(GRU(configs.RNN_NUMBER_GRU_LAYERS,
                          return_sequences = True, 
                          recurrent_dropout = configs.RNN_RECURRENT_DROPOUT))
            model.add(Dropout(dropout))
                        
        model.add(GRU(configs.RNN_NUMBER_GRU_LAYERS, 
                      recurrent_dropout = configs.RNN_RECURRENT_DROPOUT)
        )
        #model.add(Dropout(dropout))
        #model.add(Dense(configs.RNN_NUMBER_DENSE_LAYERS, 
        #                activation = configs.ACTIVATION_FUNCTION_RELU)
        #)        
        model.add(Dense(n_classes, 
                        activation = configs.ACTIVATION_FUNCTION_SOFTMAX)
        )

        #Compile Model
        model.compile(loss = configs.LOSS_SPARSE_CATEGORICAL_CROSS_ENTROPY, 
                      optimizer = configs.OPTIMIZER_ADAM, 
                      metrics = [configs.ACCURACY_METRIC])
        model.summary()
        #Return Model
        return model

    def train_model(self,train_percentage,dataset_type):
        
        path_dataset = ""
        n_classes = 0
        if(dataset_type == "R8"):
            path_dataset = configs.PATH_R8_DATASET
            n_classes = configs.NUMBER_CLASSES_R8
        elif(dataset_type == "OH"):
            path_dataset = configs.PATH_OH_DATASET
            n_classes = configs.NUMBER_CLASSES_OH
        elif(dataset_type == "R52"):
            path_dataset = configs.PATH_R52_DATASET
            n_classes = configs.NUMBER_CLASSES_R52
        else:
            raise Exception("TARGET TYPE NOT VALID")
        
        path_dataset += "complete.csv"

        helper = Helper()
        vectorizer = Vectorizer()

        #Read Dataset and split
        dataframe_train, dataframe_test = helper.read_and_split_dataset(path_dataset, train_percentage)
        
        #Build Raw dataset
        X_train_raw = dataframe_train[configs.FIELD_CSV_TEXT]
        y_train_raw = dataframe_train[configs.FIELD_CSV_INTENT]

        X_test_raw = dataframe_test[configs.FIELD_CSV_TEXT]
        y_test_raw = dataframe_test[configs.FIELD_CSV_INTENT]

        #Convert training and testing data to the correct type
        (X_train,X_test) = vectorizer.convert_phrases_to_vector(X_train_raw,X_test_raw)

        y_train = vectorizer.convert_targets_to_vector(y_train_raw,dataset_type)
        y_test = vectorizer.convert_targets_to_vector(y_test_raw,dataset_type)

        word_index = vectorizer.get_unique_tokens(X_train_raw,X_test_raw)
        embeddings_index = vectorizer.get_unique_tokens_glove()
        
        model = self.build_model(word_index,embeddings_index, n_classes)
        model.fit(X_train, y_train,
                  validation_data=(X_test, y_test),
                  epochs = configs.RNN_EPOCHS,
                  batch_size = configs.RNN_BATCH_SIZE
        )
        predicted = np.argmax(model.predict(X_test),axis = 1)
        print(predicted)        
        return metrics.classification_report(y_test, predicted)