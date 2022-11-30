from config import configs
from convolutional_neural_network.cnn_model import CNNModel

# ------------------------------------------------------
# CREATE MODEL
# ------------------------------------------------------
model = CNNModel()
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