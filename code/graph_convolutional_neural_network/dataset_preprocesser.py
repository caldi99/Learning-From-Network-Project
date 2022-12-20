import re
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.helper import Helper
from config import configs

class DatasetPreprocesser:
    #TODO : ADD COMMENTS
    def get_word_vectors(self, dataset_type):
        #Get vocabulary of the dataset
        set_vocabulary = self.get_vocabulary_dataset(dataset_type)
        
        definitions = []
        for word in set_vocabulary:
            synsets = wordnet.synsets(self.clean_phrase(word.strip()))
            
            word_defs = []
            
            for synset in synsets:
                syn_def = synset.definition()
                word_defs.append(syn_def)
            
            word_des = ' '.join(word_defs)
            if word_des == '':
                word_des = '<PAD>'
            definitions.append(word_des)

        #Create Vectorizer
        tfidf_vectorizer = TfidfVectorizer(max_features = configs.GCNN_TFID_VECTORIZER_MAX_FEATURES)

        tfidf_matrix = tfidf_vectorizer.fit_transform(definitions).toArray()

        word_vectors = []
        for i in range(len(set_vocabulary)):
            word = set_vocabulary[i]
            vector = tfidf_matrix[i]

            str_vector = []
            for j in range(len(vector)):
                str_vector.append(str(vector[j]))

            temp = ' '.join(str_vector)

            word_vector = word + ' ' + temp
            
            word_vectors.append(word_vector)
        return word_vectors

    def get_vocabulary_dataset(self, dataset_type):
        """
            This function returns the vocabulary of the dataset specified
            dataset_type :
                Type of dataset to use
            return :
                A set that corresponds to the vocabulary of the dataset
        """
        #Get cleaned version of the dataset
        phrases, classes = self.get_cleaned_dataset(dataset_type)

        #Build vocabulary
        set_vocabulary = set()
        for phrase in phrases:
            for word in phrase:
                set_vocabulary.add(word)
        return set_vocabulary
    
    def get_distinct_classes_dataset(self,dataset_type):
        """
            This function returns the distinct classes of the dataset
            dataset_type :
                Type of dataset to use
            return :
                A set that corresponds to the unique classes of the dataset
        """
        #Get cleaned version of the dataset
        phrases, classes = self.get_cleaned_dataset(dataset_type)

        #Build unique classes
        set_unique_classes = set()
        for c in classes:
            set_unique_classes.add(c)
        return set_unique_classes

    def get_cleaned_dataset(self, dataset_type):
        """
            This function return the cleaned version of the dataset provided as input
            dataset_type :
                Type of dataset to use
            return :
                Tuple that represent the cleaned dataset
        """
        #Compute path of the dataset
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
        dataframe_dataset = helper.read_dataset(path_dataset)

        #Compute phrases and classes
        phrases = dataframe_dataset[configs.FIELD_CSV_TEXT]
        classes = dataframe_dataset[configs.FIELD_CSV_INTENT]

        #Clean phrases
        list_cleaded_phrases = []
        for phrase in phrases:            
            list_cleaded_phrases.append(self.clean_phrase(phrase))
        
        #Compute word-frequency dictionary
        dictonary_cleaned_words_frequencies = self.get_word_frequencies(list_cleaded_phrases)

        #Remove not frequent words and stopwords from phrases
        return self.remove_not_frequent_words_and_stopwords_from_phrases(list_cleaded_phrases,classes,dictonary_cleaned_words_frequencies)
    
    def remove_not_frequent_words_and_stopwords_from_phrases(self, list_phrases, list_classes,dictionary_words_frequncies):
        """
            This function removes not frequent and stopwords from phrases
            list_phrases : 
                List of the phrases for which removing stopwords and not frequent words
            list_classes : 
                List of the classes of the phrases for which removing stopwords and not frequent words
            dictionary_words_frequncies : 
                Dictionary of word-frequency values for the phrases for which removing stopwords and not frequent words            
            return : 
                A couple that represent the list of phrases and their correspondance list of classes for which the stopwords and not frequent words have been removed
        """
        list_phrases_ret = []
        list_classes_ret = []
        set_stop_words = set(stopwords.words(configs.GCNN_LANGUAGE_WORDS))

        for phrase in list_phrases:
            
            #Convert the current phrase in a phrase with only the words that are (frequent) and not stop word
            phrase_words_removed = []
            for word in phrase:
                if((dictionary_words_frequncies[word] >= configs.GCNN_THRESHOLD_WORD_FREQUENCY) and (word not in set_stop_words)):
                    phrase_words_removed.append(word)
            
            #Add the phrase "transformed" to the set of phrases to return only if it contains something
            if(len(phrase_words_removed) != 0):
                index = list_phrases.index(phrase)
                list_phrases_ret.append(phrase_words_removed)
                list_classes_ret.append(list_classes[index])
        
        return list_phrases_ret, list_classes_ret

    def get_word_frequencies(self, list_phrases):
        """
            Given a list of phrases, this function return a dictionary containing for each word in those phrases the number of times it appears
            list_phrases :
                The list of phrases for which computing the dictionary of word-frequencies
            return :
                A dictionary of word-frequency values
        """
        dictionary_words_frequncies = {}        
        for phrase in list_phrases:
            for word in phrase:
                if word in dictionary_words_frequncies:
                    dictionary_words_frequncies[word] += 1
                else:
                    dictionary_words_frequncies[word] = 1        
        return dictionary_words_frequncies

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
