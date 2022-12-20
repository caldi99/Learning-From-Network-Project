import numpy as np
from dataset_preprocesser import DatasetPreprocesser

class GraphBuilder:

    def __init__(self, train_percentage):
        self.train_percentage = train_percentage
    
   

    def get_train_one_hot_encodings_labels(self, dataset_type):
        """
            This function computes the one-hot-encoding vectors for the classes of the phrases of the training set
            dataset_type : 
                Type of dataset to use
            return : 
                A matrix that represent the one-hot-encodings
        """
        #Get dataset
        dataset_preprocesser = DatasetPreprocesser()        
        phrases, classes = dataset_preprocesser.get_cleaned_dataset(dataset_type)

        #Compute training size
        train_size = len(phrases) * self.train_percentage

        #Get unique classes of the dataset
        set_unique_classes = dataset_preprocesser.get_distinct_classes_dataset(dataset_type)

        #Compute one hot encodings
        one_hot_encodings = []

        for i in range(train_size):
            #Get current class
            current_class = classes[i]
            
            #Compute one hot encoding current class
            one_hot_encoding = [0 for index in range(set_unique_classes)]
            one_hot_encoding[set_unique_classes.index(current_class)] = 1
            
            #Append current one hot encoding
            one_hot_encodings.append(one_hot_encoding)
        return np.array(one_hot_encodings)

    def get_test_one_hot_encodings_labels(self, dataset_type):
        """
            This function computes the one-hot-encoding vectors for the classes of the phrases of the test set
            dataset_type : 
                Type of dataset to use
            return : 
                A matrix that represent the one-hot-encodings
        """
        #Get dataset
        dataset_preprocesser = DatasetPreprocesser()        
        phrases, classes = dataset_preprocesser.get_cleaned_dataset(dataset_type)

        #Compute test size and train size
        train_size = len(phrases) * self.train_percentage
        test_size = len(phrases) - train_size
        
        #Get unique classes of the dataset
        set_unique_classes = dataset_preprocesser.get_distinct_classes_dataset(dataset_type)

        #Compute one hot encodings
        one_hot_encodings = []

        for i in range(test_size):
            #Get current class
            current_class = classes[train_size + i]
            
            #Compute one hot encoding current class
            one_hot_encoding = [0 for index in range(set_unique_classes)]
            one_hot_encoding[set_unique_classes.index(current_class)] = 1
            
            #Append current one hot encoding
            one_hot_encodings.append(one_hot_encoding)
        return np.array(one_hot_encodings)


     def get_training_feature_vectors(self, dataset_type):
        
        dataset_preprocesser = DatasetPreprocesser()        
        phrases, labels = dataset_preprocesser.get_cleaned_dataset(dataset_type)

    def get_test_feature_vectors(self, dataset_type):

    def get_all_feature_vectors(self, dataset_type):
    

