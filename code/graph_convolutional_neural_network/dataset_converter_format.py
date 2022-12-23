from config import configs
from utils.helper import Helper
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from scipy import sparse as sp
from math import log
import nltk
import re
import random
import numpy as np
import pickle as pkl
import os

class DatasetConverterFormat:
    def convert_dataset(self, dataset_type, train_val_percentage, train_test_percentage):  
        """
            This function convert the dataset from the cvs file into the format used by training the Graph Convolutional Neural Network
            dataset_type :
                Type of the dataset to use
            train_val_percentage : 
                Percentage of the dataset to use as training w.r.t to the part of the dataset to use as training
            train_test_percentage :
                Percentage of the dataset to use as training and validation
        """      
        #download wordnet
        nltk.download('wordnet')
        #Download stopwords
        nltk.download("stopwords")
        
        #Get Path of the Dataset
        path_dataset = ""
        if (dataset_type == "R8"):
            path_dataset = configs.PATH_R8_DATASET
        elif (dataset_type == "OH"):
            path_dataset = configs.PATH_OH_DATASET
        elif (dataset_type == "R52"):
            path_dataset = configs.PATH_R52_DATASET
        else:
            raise Exception("TARGET TYPE NOT VALID")

        path_dataset += "complete.csv"

        #Read dataset
        helper = Helper()
        dataframe_train, dataframe_test = helper.read_and_split_dataset(path_dataset,train_test_percentage)

        #Create correct paths
        
        # dataset/<dataset_type>_<train_val_percentage>.txt
        path_split = configs.GCNN_PATH_DATASET_SPLIT + dataset_type + "_" +str(train_val_percentage) + configs.GCNN_DATASET_FILE_FORMAT
        # dataset/<dataset_type>_<train_val_percentage>_shuffled.txt
        path_split_shuffled = configs.GCNN_PATH_DATASET_SPLIT + dataset_type + "_" +str(train_val_percentage) + "_shuffled" +configs.GCNN_DATASET_FILE_FORMAT

        # dataset/<dataset_type>_<train_val_percentage>_train_val_shuffled.txt
        path_split_train_val_shuffled = configs.GCNN_PATH_DATASET_SPLIT + dataset_type + "_" +str(train_val_percentage) + "_train_val_shuffled" +configs.GCNN_DATASET_FILE_FORMAT
        # dataset/<dataset_type>_<train_val_percentage>_ids_train.txt
        path_ids_path_split_lines_train_test = configs.GCNN_PATH_DATASET_SPLIT + dataset_type + "_" +str(train_val_percentage) + "_ids_train_test" +configs.GCNN_DATASET_FILE_FORMAT
        # dataset/<dataset_type>_<train_val_percentage>_ids_test.txt
        path_ids_path_split_lines_test_test = configs.GCNN_PATH_DATASET_SPLIT + dataset_type + "_" +str(train_val_percentage) + "_ids_test_test" + configs.GCNN_DATASET_FILE_FORMAT
        
        # dataset/corpus/<dataset_type>_<train_val_percentage>.txt
        path_corpus = configs.GCNN_PATH_DATASET_CORPUS + dataset_type + "_" + str(train_val_percentage) + configs.GCNN_DATASET_FILE_FORMAT
        # dataset/corpus/<dataset_type>_<train_val_percentage>_cleaned.txt
        path_corpus_cleaned = configs.GCNN_PATH_DATASET_CORPUS + dataset_type + "_" + str(train_val_percentage) + "_cleaned" + configs.GCNN_DATASET_FILE_FORMAT
        # dataset/corpus/<dataset_type>_<train_val_percentage>_train_cleaned.txt
        path_corpus_train_test_cleaned = configs.GCNN_PATH_DATASET_CORPUS + dataset_type + "_" + str(train_val_percentage) + "_train_test_cleaned" + configs.GCNN_DATASET_FILE_FORMAT
        # dataset/corpus/<dataset_type>_<train_val_percentage>_test_cleaned.txt
        path_corpus_test_test_cleaned = configs.GCNN_PATH_DATASET_CORPUS + dataset_type + "_" + str(train_val_percentage) + "_test_test_cleaned" + configs.GCNN_DATASET_FILE_FORMAT
        # dataset/corpus/<dataset_type>_<train_val_percentage>_cleaned_shuffled.txt
        path_corpus_clean_shuffled = configs.GCNN_PATH_DATASET_CORPUS + dataset_type + "_" + str(train_val_percentage) + "_cleaned_shuffled" + configs.GCNN_DATASET_FILE_FORMAT
        
        # dataset/corpus/<dataset_type>_<train_val_percentage>_labels.txt
        path_labels = configs.GCNN_PATH_DATASET_CORPUS + dataset_type + "_" + str(train_val_percentage) + "_labels" +configs.GCNN_DATASET_FILE_FORMAT

        # dataset/corpus/<dataset_type>_<train_val_percentage>_vocabulary.txt
        path_vocabulary = configs.GCNN_PATH_DATASET_CORPUS + dataset_type + "_" + str(train_val_percentage) + "_vocabulary" +configs.GCNN_DATASET_FILE_FORMAT
        # dataset/corpus/<dataset_type>_<train_val_percentage>_vocabulary_definitions.txt
        path_vocabulary_definitions = configs.GCNN_PATH_DATASET_CORPUS + dataset_type + "_" + str(train_val_percentage) + "_vocabulary_definitions" +configs.GCNN_DATASET_FILE_FORMAT
        # dataset/corpus/<dataset_type>_<train_val_percentage>_word_vectors.txt
        path_word_vectors = configs.GCNN_PATH_DATASET_CORPUS + dataset_type + "_" + str(train_val_percentage) + "_word_vector" +configs.GCNN_DATASET_FILE_FORMAT

        #Save split file
        self.save_split_file(dataframe_train,"train", 0, path_split)
        self.save_split_file(dataframe_test,"test",len(dataframe_train), path_split)

        #Compute corpus
        corpus_train_test = dataframe_train[configs.FIELD_CSV_TEXT]
        corpus_test_test = dataframe_test[configs.FIELD_CSV_TEXT]        
        corpus = list(corpus_train_test) + list(corpus_test_test)

        #Save corpus
        self.save_list(corpus, path_corpus)

        #Compute cleaned corpus
        corpus_train_test_cleaned = self.get_cleaned_corpus(corpus_train_test)
        corpus_test_test_cleaned = self.get_cleaned_corpus(corpus_test_test)
        corpus_cleaned = list(corpus_train_test) + list(corpus_test_test_cleaned)

        #Save corpus cleaned files
        self.save_list(corpus_train_test_cleaned, path_corpus_train_test_cleaned)
        self.save_list(corpus_train_test_cleaned, path_corpus_test_test_cleaned)
        self.save_list(corpus_cleaned, path_corpus_cleaned)

        #Get split lines
        all_split_lines ,train_test_split_lines, test_test_split_lines = self.get_split_lines(path_split)

        #Get ids of split lines shuffled
        ids_split_lines_shuffled_train_test = self.get_ids_split_lines_shuffled(train_test_split_lines, all_split_lines)
        ids_split_lines_shuffled_test_test = self.get_ids_split_lines_shuffled(test_test_split_lines, all_split_lines)
        ids_split_lines_shuffled_all = ids_split_lines_shuffled_train_test + ids_split_lines_shuffled_test_test

        #Save ids of split lines shuffled
        self.save_list(ids_split_lines_shuffled_train_test, path_ids_path_split_lines_train_test)
        self.save_list(ids_split_lines_shuffled_test_test, path_ids_path_split_lines_test_test)
 
        #Get shuffled version of corpus and split lines
        corpus_cleaned_shuffled = self.get_shuffled_version(corpus_cleaned, ids_split_lines_shuffled_all)
        all_split_lines_shuffled = self.get_shuffled_version(all_split_lines, ids_split_lines_shuffled_all)

        #Save corpus and all splitted lines shuffled
        self.save_list(corpus_cleaned_shuffled, path_corpus_clean_shuffled)
        self.save_list(all_split_lines_shuffled, path_split_shuffled)

        #Get the vocabulary of the (shuffled cleaned corpus)
        vocabulary = self.get_vocabulary(corpus_cleaned_shuffled)

        #Save vocabulary
        self.save_list(vocabulary,path_vocabulary)

        #Get labels
        labels = self.get_labels(all_split_lines_shuffled)

        #Save labels
        self.save_list(labels,path_labels)

        #Get vocabulary definitions
        vocabulary_definitions = self.get_vocabulary_defintions(vocabulary)

        #Save vocabulary definitions
        self.save_list(vocabulary_definitions, path_vocabulary_definitions)

        #Get word vectors
        word_vectors = self.get_word_vector(vocabulary, vocabulary_definitions)
        
        #Save word vectors
        self.save_list(word_vectors, path_word_vectors)

        #Compute train size
        train_test_size = len(ids_split_lines_shuffled_train_test)
        test_test_size = len(ids_split_lines_shuffled_test_test)
        train_val_size = int(train_test_size * train_val_percentage)

        #Compute train_val split lines
        train_val_split_lines = all_split_lines_shuffled[ : train_val_size]

        #Save train val split lines
        self.save_list(train_val_split_lines,path_split_train_val_shuffled)

        #Get word vectors map
        word_vector_map = self.get_word_vector_map(path_word_vectors)
  
        #Compute train and test feature matrix
        x_train = self.get_x_train_or_test(train_val_size, 0, corpus_cleaned_shuffled, word_vector_map)
        x_test = self.get_x_train_or_test(test_test_size, train_test_size, corpus_cleaned_shuffled, word_vector_map)
        x_all = self.get_x_all(train_test_size, vocabulary,corpus_cleaned_shuffled,word_vector_map)

        #Get one hot encoding
        y_train = self.get_y_train_or_test(train_val_size, 0,all_split_lines_shuffled, labels)
        y_test = self.get_y_train_or_test(test_test_size, train_test_size,all_split_lines_shuffled, labels)
        y_all = self.get_y_all(train_test_size, vocabulary,all_split_lines_shuffled, labels)
        
        #Compute adj
        adj = self.get_adj_matrix(corpus_cleaned_shuffled, vocabulary, train_test_size, test_test_size)

        #Paths objects to dump        
        # dataset/corpus/<dataset_type>_<train_val_percentage>_x_train.x
        path_x_train = configs.GCNN_PATH_DUMP_OBJECTS + dataset_type + "_" + str(train_val_percentage) + "_x_train.x"
        # dataset/corpus/<dataset_type>_<train_val_percentage>_x_test.x
        path_x_test = configs.GCNN_PATH_DUMP_OBJECTS + dataset_type + "_" + str(train_val_percentage) + "_x_test.x"
        # dataset/corpus/<dataset_type>_<train_val_percentage>_x_all.x
        path_x_all = configs.GCNN_PATH_DUMP_OBJECTS + dataset_type + "_" + str(train_val_percentage) + "_x_all.x"
        # dataset/corpus/<dataset_type>_<train_val_percentage>_y_train.y
        path_y_train = configs.GCNN_PATH_DUMP_OBJECTS + dataset_type + "_" + str(train_val_percentage) + "_y_train.y"
        # dataset/corpus/<dataset_type>_<train_val_percentage>_y_test.y
        path_y_test = configs.GCNN_PATH_DUMP_OBJECTS + dataset_type + "_" + str(train_val_percentage) + "_y_test.y"
        # dataset/corpus/<dataset_type>_<train_val_percentage>_y_all.y
        path_y_all = configs.GCNN_PATH_DUMP_OBJECTS + dataset_type + "_" + str(train_val_percentage) + "_y_all.y"
        # dataset/corpus/<dataset_type>_<train_val_percentage>_adj.adj
        path_adj = configs.GCNN_PATH_DUMP_OBJECTS + dataset_type + "_" + str(train_val_percentage) + "_adj.adj"

        #Dump objects
        self.dump_object(x_train,path_x_train)
        self.dump_object(x_test,path_x_test)
        self.dump_object(x_all,path_x_all)
        self.dump_object(y_train,path_y_train)
        self.dump_object(y_test,path_y_test)
        self.dump_object(y_all,path_y_all)
        self.dump_object(adj,path_adj)
        

    def save_list(self,list_to_save, path_file):
        """
            This function saves the content of a list into a file
            list_to_save :
                The list to be saved
            path_file :
                Path where to save the list
        """
        file = open(path_file, configs.FILE_OPEN_MODALITY)
        for elem in list_to_save:
            file.write("{}\n".format(elem))
        file.close()
    
    def save_split_file(self, dataframe, set_type, start_index, path_file):
        """
            This function creates a file, if does not exitsts or append to it the following content : index   [train|test]    label
            dataframe :
                The dataframe to use for writing
            set_type :
                Either "train" or "test"
            start_index :
                Initial "index" to use
            path_file :
                Path of where to save such file
        """
        file = open(path_file, configs.FILE_OPEN_MODALITY)        
        index = start_index
        for i, row in dataframe.iterrows():      
            file.write("{}\t{}\t{}\n".format(index,set_type,row[configs.FIELD_CSV_INTENT]))
            index += 1
        file.close()

    def get_labels(self, split_lines):
        """
            This function returns all the labels of the dataset
            split_lines :
                All the split lines
            return :
                The list of all the labels
        """
        label_set = set()
        for line in split_lines:
            label_set.add(line.split('\t')[2])
        return list(label_set)

    def get_vocabulary_defintions(self, vocabulary):
        """
            This function compute the defintions for the words in a vocabulary
            vocabulary :
                The vocabulary with which computing the words definitions
            return :
                The word defintions
        """
        definitions = []

        for word in vocabulary:
            synsets = wn.synsets(self.clean_phrase(word.strip()))
            word_definitions = []
            for synset in synsets:
                syn_def = synset.definition()
                word_definitions.append(syn_def)
            word_des = ' '.join(word_definitions)
            if word_des == '':
                word_des = '<PAD>'
            definitions.append(word_des)
        return definitions

    def get_word_vector(self,vocabulary, vocabulary_definitions):
        """
            This function computes the word vectors for the vocabulary of the corpus of the dataset
            vocabulary : 
                vocabulary of the corpus of the dataset
            vocabulary_definitions :
                vocabulary definitions of the corpus of the dataset
            return :
                The word vectors
        """
        #Create vectorizer
        tfidf_vec = TfidfVectorizer(max_features = configs.GCNN_WORD_EMBEDDINGS_DIM) 
        tfidf_matrix_array = tfidf_vec.fit_transform(vocabulary_definitions).toarray()
        
        #Compute word vectors
        word_vectors = []
        for i in range(len(vocabulary)):
            word = vocabulary[i]
            vector = tfidf_matrix_array[i]
            str_vector = []
            for j in range(len(vector)):
                str_vector.append(str(vector[j]))
            temp = ' '.join(str_vector)
            word_vector = word + ' ' + temp
            word_vectors.append(word_vector)
        return word_vectors

    def get_word_vector_map(self, path_word_vector):
        """
            This function return a dictionary containing word-vector couples
            path_word_vector :
                Path where the word vector file is
            return :
                A dictionary containing word-vector couples
        """
        word_vector_map = {}
        with open(path_word_vector) as file:
            for line in file.readlines():
                row = line.strip().split(' ')
                if(len(row) > 2):
                    vector = row[1:]                    
                    for i in range(len(vector)):
                        vector[i] = float(vector[i])
                    word_vector_map[row[0]] = vector
        return word_vector_map

    def get_vocabulary(self, corpus):
        """
            This function return the vocabulary of the provided corpus
            corpus : 
                The corpus to use to build the vocabulary
            return :
                A list that represent the corpus
        """
        word_set = set()
        for phrase in corpus:
            for word in phrase.split():
                word_set.add(word)
        return list(word_set)

    def get_cleaned_corpus(self,corpus):
        """
            This function return the corpus of a dataset without words that are stopwords and are not frequent
            corpus :
                The corpus of a dataset
            return :
                The corpus of a dataset without words that are stopwords and are not frequent
        """
        #Get english stopwords
        stop_words = set(stopwords.words(configs.GCNN_LANGUAGE_WORDS))

        #Get the word-count dictionary
        word_count = self.get_word_count_dictionary(corpus)

        #Clean phrases
        cleaned_corpus = []
        for phrase in corpus:
            words = self.clean_phrase(phrase).split()
            phrase_words = []
            for word in words:
                if word not in stop_words and word_count[word] >= configs.GCNN_THRESHOLD_WORD_FREQUENCY:
                    phrase_words.append(word)
            cleaned_corpus.append(' '.join(phrase_words).strip())
        
        return cleaned_corpus

    def get_word_count_dictionary(self, corpus):
        """
            This function returns a dictionary of word-count instances
            corpus :
                The corpus of the dataset
            return :
                A dictionary of word-count elements
        """
        word_count = {}

        for phrase in corpus:
            words = self.clean_phrase(phrase).split()
            for word in words:
                if word in word_count:
                    word_count[word] += 1
                else:
                    word_count[word] = 1
        return word_count
    
    def get_ids_split_lines_shuffled(self, list, all_list):
        """
            This function returns the ids associated to each line of list with respect to all_list, randomized
            all_list : 
                The list of all lines inside the split file
            return :
                The list of ids of lines of list in all_list
        """
        #Compute ids
        phrases_ids = []
        for line in list:
            phrases_ids.append(all_list.index(line))

        #Apply Randomization
        random.shuffle(phrases_ids)

        return phrases_ids

    def get_split_lines(self,path_split):               
        """
            This function returns the lines of the split file into 3 lists : all the lines, only the train lines, only the test lines
            path_split : 
                Path of the split file
            return :
                A triple of lists : all the lines, only the train lines, only the test lines of the split file
        """
        #Compute the lists
        all_list = []
        train_list = []
        test_list = []
        with open(path_split) as file:
            lines = file.readlines()
            for line in lines:
                all_list.append(line.strip())
                if line.split("\t")[1].find('test') != -1:
                    test_list.append(line.strip())
                elif line.split("\t")[1].find('train') != -1:
                    train_list.append(line.strip())

        return all_list,train_list,test_list

    def get_shuffled_version(self, list_to_be_shuffled, ids_shuffled):
        """
            This function return the suffled version of a list according to the ids that have been shuffled
            list_to_be_shuffled :
                List that must be shuffled
            ids_shuffled :
                Ids to use to shuffle the list
            return :
                The shuffled version of the list
        """
        list_shuffled = []
        for id in ids_shuffled:
            list_shuffled.append(list_to_be_shuffled[id])
        
        return list_shuffled
    
    def clean_phrase(self, phrase):
        """
            This function clean a phrase as done here : https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
            phrase :   
                The phrase to be cleaned
            return :
                The cleaned version of a phrase
        """
        phrase = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", phrase)
        phrase = re.sub(r"\'s", " \'s", phrase)
        phrase = re.sub(r"\'ve", " \'ve", phrase)
        phrase = re.sub(r"n\'t", " n\'t", phrase)
        phrase = re.sub(r"\'re", " \'re", phrase)
        phrase = re.sub(r"\'d", " \'d", phrase)
        phrase = re.sub(r"\'ll", " \'ll", phrase)
        phrase = re.sub(r",", " , ", phrase)
        phrase = re.sub(r"!", " ! ", phrase)
        phrase = re.sub(r"\(", " \( ", phrase)
        phrase = re.sub(r"\)", " \) ", phrase)
        phrase = re.sub(r"\?", " \? ", phrase)
        phrase = re.sub(r"\s{2,}", " ", phrase)
        return phrase.strip().lower()
    
    def get_x_train_or_test(self, size, offset, corpus_cleaned_shuffled, word_vector_map):
        """
            This function is used to compute the feature vectors for training and testing
            size :
                The size of the list of train/test phrases
            offset :
                Offset to use for accessing to corpus_cleaned_shuffled
            corpus_cleaned_shuffled :
                The corpus cleaned shuffled of the dataset
            word_vector_map :
                The word vector map
            return :
                A sparse matrix containing the data
        """
        row = []
        col = []
        data = []

        for i in range(size):
            phrase_vec = np.array([0.0 for k in range(configs.GCNN_WORD_EMBEDDINGS_DIM)])
            phrase = corpus_cleaned_shuffled[offset + i]
            for word in phrase:
                if word in word_vector_map:
                    word_vector = word_vector_map[word]            
                    phrase_vec = phrase_vec + np.array(word_vector)                    
            
            for j in range(configs.GCNN_WORD_EMBEDDINGS_DIM):
                row.append(i)
                col.append(j)                                     
                data.append(phrase_vec[j] / len(phrase))

        return sp.csr_matrix((data, (row, col)), shape=(size, configs.GCNN_WORD_EMBEDDINGS_DIM))

    def get_x_all(self, train_size, vocabulary, corpus_cleaned_shuffled, word_vector_map):
        """
            This function is used to compute the feature vectors of both labeled and unlabeled training
            train_size :
                The size of the list of training phrases
            vocabulary :
                The vocabulary of the phrases of the dataset
            corpus_cleaned_shuffled :
                The corpus cleaned shuffled of the dataset
            word_vector_map :
                The word vector map
            return :
                A sparse matrix containing the data
        """

        #Compute word vectors
        word_vectors = np.random.uniform(-0.01, 0.01, (len(vocabulary), configs.GCNN_WORD_EMBEDDINGS_DIM))
        for i in range(len(vocabulary)):
            word = vocabulary[i]
            if word in word_vector_map:
                vector = word_vector_map[word]
                word_vectors[i] = vector

        #Compute sparse feature matrix
        row = []
        col = []
        data = []

        for i in range(train_size):
            phrase_vec = np.array([0.0 for k in range(configs.GCNN_WORD_EMBEDDINGS_DIM)])
            phrase = corpus_cleaned_shuffled[i]
            for word in phrase:
                if word in word_vector_map:
                    word_vector = word_vector_map[word]            
                    phrase_vec = phrase_vec + np.array(word_vector)                    
            
            for j in range(configs.GCNN_WORD_EMBEDDINGS_DIM):
                row.append(i)
                col.append(j)                                     
                data.append(phrase_vec[j] / len(phrase))


        for i in range(len(vocabulary)):
            for j in range(configs.GCNN_WORD_EMBEDDINGS_DIM):
                row.append(int(i + train_size))
                col.append(j)
                data.append(word_vectors.item((i, j)))
        
        #Convert to numpy array
        row = np.array(row)
        col = np.array(col)
        data = np.array(data)

        return sp.csr_matrix((data, (row, col)), shape=(train_size + len(vocabulary), configs.GCNN_WORD_EMBEDDINGS_DIM))  

    def get_y_train_or_test(self, size, offset, all_split_lines_shuffled, label_list):
        """
            This function is used to compute the feature vectors for training and testing
            size :
                The size of the list of train/test phrases
            offset :
                Offset to use for accessing to all_split_lines_shuffled
            all_split_lines_shuffled :
                All lines of the splitted file shuffled
            label_list :
                The list of labels
            return :
                A matrix of one hot encoding of the labels
        """
        y = []
        for i in range(size):
            doc_meta = all_split_lines_shuffled[offset + i]
            one_hot = [0 for l in range(len(label_list))]
            one_hot[label_list.index(doc_meta.split('\t')[2])] = 1
            y.append(one_hot)
        
        return np.array(y)
    
    def get_y_all(self, train_size, vocabulary, all_split_lines_shuffled, label_list):
        """
            This function is used to compute the feature vectors of both labeled and unlabeled training
            train_size :
                The size of the list of training phrases
            vocabulary :
                The vocabulary of the phrases of the dataset
            all_split_lines_shuffled :
                All lines of the splitted file shuffled
            label_list :
                The list of labels
            return :
                A matrix of one hot encoding of the labels
        """

        y = []
        for i in range(train_size):
            doc_meta = all_split_lines_shuffled[i]
            one_hot = [0 for l in range(len(label_list))]
            one_hot[label_list.index(doc_meta.split('\t')[2])] = 1
            y.append(one_hot)

        for i in range(len(vocabulary)):
            one_hot = [0 for l in range(len(label_list))]
            y.append(one_hot)

        return np.array(y) 

    def get_windows(self,corpus_cleaned_shuffled):
        """
            Return all the possible windows (phrases with a limited length) that can be built with the phrases of the corpus of the dataset
            corpus_cleaned_shuffled :
                The corpus of the dataset
            return :
                The windows
        """
        windows = []
        for phrase in corpus_cleaned_shuffled:
            words = phrase.split()
            length = len(words)
            if length <= configs.GCNN_WINDOW_SIZE:
                windows.append(words)
            else:
                for j in range(length - configs.GCNN_WINDOW_SIZE + 1):
                    windows.append(words[j: j + configs.GCNN_WINDOW_SIZE])
        return windows

    def get_window_frequency_map(self, windows):
        """
            This function compute the window-frequency map
            windows :
                Windows to use to compute the window-frequency map
            return :
                The window frequency map
        """
        window_frequency_map = {}
        for window in windows:
            appeared = set()
            for i in range(len(window)):
                if window[i] in appeared:
                    continue
                if window[i] in window_frequency_map:
                    window_frequency_map[window[i]] += 1
                else:
                    window_frequency_map[window[i]] = 1
                appeared.add(window[i])
        return window_frequency_map

    def get_vocabulary_word_id_map(self,vocabulary):
        """
            This function computes the mapping between a word in the vocabulary and the corresponding id
            vocabulary :
                The vocabulary of the dataset
            return :
                The word-id map
        """
        word_id_map = {}
        for i in range(len(vocabulary)):
            word_id_map[vocabulary[i]] = i
        return word_id_map

    def get_word_pair_count_map(self, windows, word_id_map):
        """
            This function compute the word_pair-count map
            windows :
                The windows of the phrases of the corpus
            word_id_map :
                The word-id map
            return :
                The word_pair-count map
        """
        word_pair_count = {}
        for window in windows:
            for i in range(1, len(window)):
                for j in range(0, i):
                    word_i = window[i]
                    word_i_id = word_id_map[word_i]
                    word_j = window[j]
                    word_j_id = word_id_map[word_j]
                    
                    if word_i_id == word_j_id: # pair(i,i)
                        continue
                    # (i,j) pair
                    word_pair_str = str(word_i_id) + ',' + str(word_j_id) 
                    if word_pair_str in word_pair_count:
                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1
                    # (j,i) pair
                    word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                    if word_pair_str in word_pair_count:
                        word_pair_count[word_pair_str] += 1
                    else:
                        word_pair_count[word_pair_str] = 1
        return word_pair_count

    def get_phrase_word_pair_frequency(self,corpus_cleaned_shuffled, word_id_map):
        """
            This function compute the phrase-word_pair-frequency map
            corpus_cleaned_shuffled :
                The corpus of the dataset
            word_id_map :
                The word-id map
            return :
                The phrase-word_pair-frequency map
        """
        phraseword_pair_frequency = {}

        for index in range(len(corpus_cleaned_shuffled)):
            phrase = corpus_cleaned_shuffled[index]
            words = phrase.split()
            for word in words:
                word_id = word_id_map[word]
                phrase_word_str = str(index) + ',' + str(word_id)
                if phrase_word_str in phraseword_pair_frequency:
                    phraseword_pair_frequency[phrase_word_str] += 1
                else:
                    phraseword_pair_frequency[phrase_word_str] = 1
        
        return phraseword_pair_frequency

    def get_word_phrases_list_map(self,corpus_cleaned_shuffled):
        """
            This function computes the word-phrases_list-pair map i.e. for each word we compute the indexes of the phrases where it appears in the corpus
            corpus_cleaned_shuffled :
                The corpus of the dataset
            return :
                The word-phrases_list-pair map
        """
        word_phrases_list_map = {}

        for i in range(len(corpus_cleaned_shuffled)):
            phrase = corpus_cleaned_shuffled[i]
            words = phrase.split()
            appeared = set()
            for word in words:
                if word in appeared:
                    continue
                if word in word_phrases_list_map:
                    phrases_list = word_phrases_list_map[word]
                    phrases_list.append(i)
                    word_phrases_list_map[word] = phrases_list
                else:
                    word_phrases_list_map[word] = [i]
                appeared.add(word)
        
        return word_phrases_list_map

    def get_word_presence_frequency_map(self, word_phrases_map):
        """
            This function computes the word_presence_frequency_map i.e. in how many phrases a word appears for each word
            word_phrases_map :
                The word_phrases_map map
            return :
                The word_presence_frequency_map
        """
        word_presence_frequency_map = {}
        for word, phrases_appearance in word_phrases_map.items():
            word_presence_frequency_map[word] = len(phrases_appearance)
        return word_presence_frequency_map

    def get_adj_matrix(self, corpus_cleaned_shuffled, vocabulary, train_size, test_size):   
        """
            This function computes the adjacency matrix for the graph
            corpus_cleaned_shuffled : 
                The corpus of the dataset
            vocabulary :
                The vocabulary of the corpus of the dataset
            train_size :
                Training size
            test_size :
                Test size
            return :
                The adjacency matrix of the graph
        """
        #Compute windows     
        windows = self.get_windows(corpus_cleaned_shuffled)
        #Compute window_frequency_map
        window_frequency_map = self.get_window_frequency_map(windows)

        #Compute word_id_map
        word_id_map = self.get_vocabulary_word_id_map(vocabulary)

        #Compute wordpair_count_map
        wordpair_count_map = self.get_word_pair_count_map(windows,word_id_map)

        # compute phraseword_pair_frequency_map
        phraseword_pair_frequency_map = self.get_phrase_word_pair_frequency(corpus_cleaned_shuffled,word_id_map)

        #Compute word_presence_frequency_map
        word_presence_frequency_map = self.get_word_presence_frequency_map(self.get_word_phrases_list_map(corpus_cleaned_shuffled))
        row = []
        col = []
        weight = []

        num_window = len(windows)
        for key in wordpair_count_map:
            temp = key.split(',')
            i = int(temp[0])
            j = int(temp[1])
            count = wordpair_count_map[key]
            word_freq_i = window_frequency_map[vocabulary[i]]
            word_freq_j = window_frequency_map[vocabulary[j]]
            pmi = log((1.0 * count / num_window) / (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
            if pmi <= 0:
                continue
            row.append(train_size + i)
            col.append(train_size + j)
            weight.append(pmi)

        for i in range(len(corpus_cleaned_shuffled)):
            doc_words = corpus_cleaned_shuffled[i]
            words = doc_words.split()
            doc_word_set = set()
            for word in words:
                if word in doc_word_set:
                    continue
                j = word_id_map[word]
                key = str(i) + ',' + str(j)
                freq = phraseword_pair_frequency_map[key]
                if i < train_size:
                    row.append(i)
                else:
                    row.append(i + len(vocabulary))
                col.append(train_size + j)
                idf = log(1.0 * len(corpus_cleaned_shuffled) / word_presence_frequency_map[vocabulary[j]])
                weight.append(freq * idf)
                doc_word_set.add(word)

        node_size = train_size + len(vocabulary) + test_size
        return sp.csr_matrix((weight, (row, col)), shape=(node_size, node_size))

    def dump_object(self, object, path_file):
        with open(path_file, 'wb') as file:
            pkl.dump(object, file)