from config import configs
from convolutional_neural_network.cnn_model import CNNModel

# ------------------------------------------------------
# CREATE MODEL
# ------------------------------------------------------
model = CNNModel()
# ------------------------------------------------------
# TRAIN MODEL
# ------------------------------------------------------
train_val_percentage = 0.8  #only this to change (values = [0.7,0.8])
train_test_percentage = 0.7
dataset_type = "R8"

results = model.train_model(train_test_percentage,train_val_percentage,dataset_type)

# ------------------------------------------------------
# RESULTS
# ------------------------------------------------------
print("RESULTS : ")
print(results)