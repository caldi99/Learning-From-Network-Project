from config import configs
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class Vectorizer:

    def convert_targets_to_vector(self, list_targets : list, dataset_type : str) -> list:
        """
            This function converts the list of target words into a integer vector
            list_targets : 
                List of target words
            dataset_type :
                The type of dataset
            return :
                List of integers that represents targets
        """

        #Get the list of unique classes for the specified dataset
        classes_dataset = []        
        if(dataset_type == "R8"):
            classes_dataset = configs.CLASSES_R8_DATASET
        elif(dataset_type == "OH"):
            classes_dataset = configs.CLASSES_OH_DATASET
        elif(dataset_type == "R52"):
            classes_dataset = configs.CLASSES_R52_DATASET
        else:
            raise Exception("TARGET TYPE NOT VALID")

        #Build the list of integers that converts the list of targets
        ret_list = []
        for word in list_targets:            
            ret_list.append(classes_dataset.index(word))

        return np.array(ret_list)
    
    def convert_phrases_to_vector(self,phrases_train : list, phrases_val : list, phrases_test : list) -> tuple:
        """
            This function convert train phrases and test phrases into vectors
            phrases_train :
                The list of training phareses to convert
            phrases_val :
                The list of validation phrases to convert
            phrases_test :
                The list of test phrases to convert
            return :
                A tuple that represent the embeddings for training validation and test phrases
        """
        union = np.array(np.concatenate((phrases_train, phrases_val, phrases_test), axis = 0))
        tokenizer = Tokenizer(num_words = configs.MAXIMUM_NUMBER_WORDS)
        tokenizer.fit_on_texts(union)
        union = pad_sequences(tokenizer.texts_to_sequences(union), 
                            maxlen = configs.MAXIMUM_NUMBER_WORDS)        
        union = union[np.arange(union.shape[0])]
        train = union[0 : len(phrases_train) , ]
        val = union[len(phrases_train) : len(phrases_train) + len(phrases_val),]
        test = union[len(phrases_train) + len(phrases_val) : ,]

        return (train, val, test)
    
    def get_unique_tokens(self,phrases_train : list, phrases_val : list) -> dict:
        """
            This function returns the unique tokens inside the train and test phrases
            phrases_train :
                The list of training phrases 
            phrases_val :
                The list of test phrases to convert
            return :
                A dictionary containing the unique tokens with their coresponding positions
        """

        union = np.array(np.concatenate((phrases_train,phrases_val), axis = 0))
        tokenizer = Tokenizer(num_words = configs.MAXIMUM_NUMBER_WORDS)
        tokenizer.fit_on_texts(union)

        return tokenizer.word_index
    
    def get_unique_tokens_glove(self) -> dict:
        """
            This function returns the unique tokens inside the glove.6B-50d.txt text file            
            return :
                A dictionary containing the unique tokens with their coresponding positions
        """
        ret = {}
        file = open(configs.PATH_GLOVE_6B_50D, encoding = configs.ENCODING_TYPE)
        for line in file:
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype = configs.FLOAT_32 )
            except:
                pass
            ret[word] = coefs
        file.close()

        return ret
