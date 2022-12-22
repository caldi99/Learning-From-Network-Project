from graph_convolutional_neural_network.dataset_converter_format import DatasetConverterFormat
from graph_convolutional_neural_network.gcnn_model import GCNNModel


train_val_percentage = 0.8  #only this to change (values = [0.7,0.8])
train_test_percentage = 0.7
dataset_type = "R8"

#Create datasets
d = DatasetConverterFormat()
d.convert_dataset(dataset_type,train_val_percentage,train_test_percentage) 

#Train Model
model = GCNNModel()
model.train_model(dataset_type,train_val_percentage)