
from sklearn.svm import LinearSVC
from sklearn import metrics

from config import configs
from utils.helper import Helper
from utils.glove_vectorizer import GloveVectorizer


class SVMModel:
    def train_model(self,train_test_percentage, train_val_percentage, dataset_type):
        """
            This function trains the SVM model and provide the results
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
        if(dataset_type == "R8"):
            path_dataset = configs.PATH_R8_DATASET
        elif(dataset_type == "OH"):
            path_dataset = configs.PATH_OH_DATASET
        elif(dataset_type == "R52"):
            path_dataset = configs.PATH_R52_DATASET
        else:
            raise Exception("TARGET TYPE NOT VALID")
        
        path_dataset += "complete.csv"

        helper = Helper()
        vectorizer = GloveVectorizer()

        #Read Dataset and split
        dataframe_train_test, dataframe_test_test = helper.read_and_split_dataset(path_dataset, train_test_percentage)

        #Split into validation and training
        X_train_val_raw = dataframe_train_test[configs.FIELD_CSV_TEXT]
        y_train_val = dataframe_train_test[configs.FIELD_CSV_INTENT]

        #Compute size train and validation
        train_size = int(len(X_train_val_raw) * train_val_percentage)
        val_size = len(X_train_val_raw) - train_size

        #Compute Raw dataset
        X_train_raw = X_train_val_raw[:train_size]
        X_val_raw = X_train_val_raw[train_size:]
        X_test_raw = dataframe_test_test[configs.FIELD_CSV_TEXT]
        
        #Convert training and testing data to the correct type
        X_train = vectorizer.transform(X_train_raw)
        y_train = y_train_val[:train_size]
        X_val = vectorizer.transform(X_val_raw)
        y_val = y_train_val[train_size:]
        X_test = vectorizer.transform(X_test_raw)
        y_test = dataframe_test_test[configs.FIELD_CSV_INTENT]

        #Create Model
        model = LinearSVC()

        #Fit Model
        model.fit(X_train, y_train)

        #Compute Predictions
        predicted = model.predict(X_test)
        return metrics.classification_report(y_test, predicted)
        