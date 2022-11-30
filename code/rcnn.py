from config import configs
from recurrent_convolutional_neural_network.rcnn_model import RCNNModel

# ------------------------------------------------------
# CREATE MODEL
# ------------------------------------------------------
model = RCNNModel()

# ------------------------------------------------------
# TRAIN MODEL
# ------------------------------------------------------
#results = model.train_model(0.2,"R8")
#results = model.train_model(0.2,"OH")
results = model.train_model(0.2,"R52")

# ------------------------------------------------------
# RESULTS
# ------------------------------------------------------
print("RESULTS : ")
print(results)