from config import configs
from recurrent_convolutional_neural_network.rcnn_model import RCNNModel

# ------------------------------------------------------
# CREATE MODEL
# ------------------------------------------------------
model = RCNNModel()

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