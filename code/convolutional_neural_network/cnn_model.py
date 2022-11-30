import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Input
from keras.layers import Conv1D
from keras.layers import MaxPool1D
from keras.layers import Flatten
from keras.layers.merging import Concatenate
from sklearn import metrics
import pandas as pd

from config import configs
from utils.helper import Helper
from utils.vectorizer import Vectorizer

class CNNModel:

    def build_model(self, word_index, embeddings_index, n_classes, EMBEDDING_DIM = 50, dropout = 0.5):

        embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                if len(embedding_matrix[i]) != len(embedding_vector):
                    exit(1)
                embedding_matrix[i] = embedding_vector

        # Embedding layer
        embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights = [embedding_matrix],
                            input_length = configs.MAXIMUM_NUMBER_WORDS,
                            trainable = True)

        # Variable filter size
        convs = []
        filter_sizes = []
        for filter_size in range(0, configs.CNN_NUMBER_HIDDEN_LAYERS):
            filter_sizes.append((filter_size+2))

        # Add layers
        sequence_input = Input(shape = (configs.MAXIMUM_NUMBER_WORDS, ),
                        dtype = configs.FLOAT_32)

        embedded_sequences = embedding_layer(sequence_input)

        for fsz in filter_sizes:
            l_conv = Conv1D(configs.CNN_NUMBER_FILTERS,
                             kernel_size=fsz,
                             activation=configs.ACTIVATION_FUNCTION_RELU)(embedded_sequences)
            l_pool = MaxPool1D(configs.CNN_MAX_POOLING_SIZE_FIRST_LAYER)(l_conv)
            convs.append(l_pool)

        l_merge = Concatenate(axis = 1)(convs)

        # 1st
        l_conv1 = Conv1D(configs.CNN_NUMBER_FILTERS,
                             kernel_size = configs.CNN_KERNEL_SIZE,
                             activation = configs.ACTIVATION_FUNCTION_RELU)(l_merge)
        l_conv1 = Dropout(dropout)(l_conv1)
        l_pool1 = MaxPool1D(configs.CNN_MAX_POOLING_SIZE_FIRST_LAYER)(l_conv1)

        # 2nd
        l_conv2 = Conv1D(configs.CNN_NUMBER_FILTERS,
                         kernel_size = configs.CNN_KERNEL_SIZE,
                         activation = configs.ACTIVATION_FUNCTION_RELU)(l_pool1)
        l_conv2 = Dropout(dropout)(l_conv2)
        l_pool2 = MaxPool1D(configs.CNN_MAX_POOLING_SIZE_FIRST_LAYER)(l_conv2)

        # Flatten Layer
        l_flat = Flatten()(l_pool2)

        # Dense Layers
        l_dense = Dense(configs.CNN_NUMBER_DENSE_LAYER_FIRST,
                        activation = configs.ACTIVATION_FUNCTION_RELU)(l_flat)
        l_dense = Dropout(dropout)(l_dense)
        l_dense = Dense(configs.CNN_NUMBER_DENSE_LAYER_SECOND,
                        activation = configs.ACTIVATION_FUNCTION_RELU)(l_dense)
        l_dense = Dropout(dropout)(l_dense)

        # Predictions
        preds = Dense(n_classes,
                      activation = configs.ACTIVATION_FUNCTION_SOFTMAX)(l_dense)

        model = Model(sequence_input, preds)

        # Compile model
        model.compile(loss=configs.LOSS_SPARSE_CATEGORICAL_CROSS_ENTROPY,
                      optimizer=configs.OPTIMIZER_ADAM,
                      metrics=[configs.ACCURACY_METRIC])

        model.summary()

        # Return Model
        return model

    def train_model(self, train_percentage, dataset_type):
        """
            This function trains the Recurrent Neural Network model and provide the results
            train_percentage :
                Percentage of data to use for training
            dataset_type :
                Type of the dataset to use
            return :
                The classification report obtained after training
        """

        path_dataset = ""
        n_classes = 0
        if (dataset_type == "R8"):
            path_dataset = configs.PATH_R8_DATASET
            n_classes = configs.NUMBER_CLASSES_R8
        elif (dataset_type == "OH"):
            path_dataset = configs.PATH_OH_DATASET
            n_classes = configs.NUMBER_CLASSES_OH
        elif (dataset_type == "R52"):
            path_dataset = configs.PATH_R52_DATASET
            n_classes = configs.NUMBER_CLASSES_R52
        else:
            raise Exception("TARGET TYPE NOT VALID")

        path_dataset += "complete.csv"

        helper = Helper()
        vectorizer = Vectorizer()

        # Read Dataset and split
        dataframe_train, dataframe_test = helper.read_and_split_dataset(path_dataset, train_percentage)

        # Build Raw dataset
        X_train_raw = dataframe_train[configs.FIELD_CSV_TEXT]
        y_train_raw = dataframe_train[configs.FIELD_CSV_INTENT]

        X_test_raw = dataframe_test[configs.FIELD_CSV_TEXT]
        y_test_raw = dataframe_test[configs.FIELD_CSV_INTENT]

        # Convert training and testing data to the correct type
        (X_train, X_test) = vectorizer.convert_phrases_to_vector(X_train_raw, X_test_raw)

        y_train = vectorizer.convert_targets_to_vector(y_train_raw, dataset_type)
        y_test = vectorizer.convert_targets_to_vector(y_test_raw, dataset_type)

        word_index = vectorizer.get_unique_tokens(X_train_raw, X_test_raw)
        embeddings_index = vectorizer.get_unique_tokens_glove()

        # Build Model
        model = self.build_model(word_index, embeddings_index, n_classes)

        # Train Model
        model.fit(X_train, y_train,
                  validation_data=(X_test, y_test),
                  epochs=configs.CNN_EPOCHS,
                  batch_size=configs.CNN_BATCH_SIZE
                  )

        # Compute Predictions
        predicted = np.argmax(model.predict(X_test), axis=1)
        print(predicted)
        return metrics.classification_report(y_test, predicted)
