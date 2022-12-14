from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import GRU
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.callbacks import EarlyStopping
import numpy as np
from sklearn import metrics

from config import configs
from utils.helper import Helper
from utils.vectorizer import Vectorizer


class RCNNModel:
    def build_model(self, word_index, embeddings_index, n_classes, EMBEDDING_DIM=50):
        model = Sequential()

        embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                if len(embedding_matrix[i]) != len(embedding_vector):
                    exit(1)
                embedding_matrix[i] = embedding_vector

        # Add layers

        # Embedding
        model.add(Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=configs.MAXIMUM_NUMBER_WORDS,
                            trainable=True))

        # Dropout
        model.add(Dropout(configs.RCNN_DROPOUT))

        # Conv1D - MaxPooling1D
        model.add(Conv1D(configs.RCNN_NUMBER_FILTERS,
                         configs.RCNN_KERNEL_SIZE,
                         activation=configs.ACTIVATION_FUNCTION_RELU))
        model.add(MaxPooling1D(pool_size=configs.RCNN_POOL_SIZE))
        model.add(Conv1D(configs.RCNN_NUMBER_FILTERS,
                         configs.RCNN_KERNEL_SIZE,
                         activation=configs.ACTIVATION_FUNCTION_RELU))
        model.add(MaxPooling1D(pool_size=configs.RCNN_POOL_SIZE))
        model.add(Conv1D(configs.RCNN_NUMBER_FILTERS,
                         configs.RCNN_KERNEL_SIZE,
                         activation=configs.ACTIVATION_FUNCTION_RELU))
        model.add(MaxPooling1D(pool_size=configs.RCNN_POOL_SIZE))
        model.add(Conv1D(configs.RCNN_NUMBER_FILTERS,
                         configs.RCNN_KERNEL_SIZE,
                         activation=configs.ACTIVATION_FUNCTION_RELU))
        model.add(MaxPooling1D(pool_size=configs.RCNN_POOL_SIZE))

        # LSTM
        model.add(LSTM(configs.RCNN_NUMBER_GRU_LAYERS,
                       return_sequences=True,
                       recurrent_dropout=configs.RCNN_RECURRENT_DROPOUT))
        model.add(LSTM(configs.RCNN_NUMBER_GRU_LAYERS,
                       return_sequences=True,
                       recurrent_dropout=configs.RCNN_RECURRENT_DROPOUT))
        model.add(LSTM(configs.RCNN_NUMBER_GRU_LAYERS,
                       return_sequences=True,
                       recurrent_dropout=configs.RCNN_RECURRENT_DROPOUT))
        model.add(LSTM(configs.RCNN_NUMBER_GRU_LAYERS,
                       recurrent_dropout=configs.RCNN_RECURRENT_DROPOUT))

        # Dense
        model.add(Dense(configs.RCNN_NUMBER_DENSE_LAYERS,
                        configs.ACTIVATION_FUNCTION_RELU))
        model.add(Dense(n_classes,
                        activation=configs.ACTIVATION_FUNCTION_SOFTMAX))

        # Compile Model
        model.compile(loss=configs.LOSS_SPARSE_CATEGORICAL_CROSS_ENTROPY,
                      optimizer=configs.OPTIMIZER_ADAM,
                      metrics=[configs.ACCURACY_METRIC]
        )

        model.summary()

        return model

    def train_model(self, train_test_percentage, train_val_percentage, dataset_type):
        """
            This function trains the Recurrent Convolutional Neural Network model and provide the results
            train_test_percentage :
                Percentage of data to use for training w.r.t. whole dataset
            train_val_percentage :
                Percentage of data to use for validation w.r.t. trainining + validataion set
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

        #Read Dataset and split
        dataframe_train_test, dataframe_test_test = helper.read_and_split_dataset(path_dataset, train_test_percentage)

        #Split into validation and training
        X_train_val_raw = dataframe_train_test[configs.FIELD_CSV_TEXT]
        y_train_val_raw = dataframe_train_test[configs.FIELD_CSV_INTENT]

        #Compute size train and validation
        train_size = int(len(X_train_val_raw) * train_val_percentage)
        val_size = len(X_train_val_raw) - train_size

        #Compute Raw dataset
        X_train_raw = X_train_val_raw[:train_size]
        X_val_raw = X_train_val_raw[train_size:]
        X_test_raw = dataframe_test_test[configs.FIELD_CSV_TEXT]
        
        y_train_raw = y_train_val_raw[:train_size]
        y_val_raw = y_train_val_raw[train_size:]
        y_test_raw = dataframe_test_test[configs.FIELD_CSV_INTENT]

        # Convert training and testing data to the correct type
        (X_train, X_val, X_test) = vectorizer.convert_phrases_to_vector(X_train_raw, X_val_raw, X_test_raw)

        y_train = vectorizer.convert_targets_to_vector(y_train_raw, dataset_type)
        y_val = vectorizer.convert_targets_to_vector(y_val_raw, dataset_type)
        y_test = vectorizer.convert_targets_to_vector(y_test_raw, dataset_type)

        word_index = vectorizer.get_unique_tokens(X_train_raw, X_val_raw)
        embeddings_index = vectorizer.get_unique_tokens_glove()

        # Build Model
        model = self.build_model(word_index, embeddings_index, n_classes)

        #Define callback
        callback = EarlyStopping(
            mode='min',
            monitor='val_loss',
            patience=10
        )

        # Train Model
        model.fit(X_train, y_train,
                  validation_data=(X_val, y_val),
                  epochs=configs.RCNN_EPOCHS,
                  batch_size=configs.RCNN_BATCH_SIZE,
                  callbacks=[callback]
        )

        # Compute Predictions
        predicted = np.argmax(model.predict(X_test), axis=1)
        print(predicted)
        return metrics.classification_report(y_test, predicted)
