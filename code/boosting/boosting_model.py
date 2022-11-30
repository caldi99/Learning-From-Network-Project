
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics

from config import configs
from utils.helper import Helper
from utils.glove_vectorizer import GloveVectorizer


class BoostingModel:
    def train_model(self,train_percentage,dataset_type):
        """
            This function trains the Boosting model and provide the results
            train_percentage :
                Percentage of data to use for training
            dataset_type :
                Type of the dataset to use
            return :
                The classification report obtained after training
        """
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
        vectorizer = GloveVectorizer()

        #Read Dataset and split
        dataframe_train, dataframe_test = helper.read_and_split_dataset(path_dataset, train_percentage)

        #Build Raw dataset
        X_train_raw = dataframe_train[configs.FIELD_CSV_TEXT]
        X_test_raw = dataframe_test[configs.FIELD_CSV_TEXT]
        
        #Convert training and testing data to the correct type
        X_train = vectorizer.transform(X_train_raw)
        y_train = dataframe_train[configs.FIELD_CSV_INTENT]
        X_test = vectorizer.transform(X_test_raw)
        y_test = dataframe_test[configs.FIELD_CSV_INTENT]

        #Create Model
        model = GradientBoostingClassifier(n_estimators = configs.BOOSTING_NUMBER_ESTIMATORS)

        #Fit Model
        model.fit(X_train, y_train)

        #Compute Predictions
        predicted = model.predict(X_test)
        return metrics.classification_report(y_test, predicted)
        