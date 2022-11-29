

from config import configs
from recurrent_neural_network.rnn_model import RNNModel

# ------------------------------------------------------
# CREATE MODEL
# ------------------------------------------------------
model = RNNModel()

# ------------------------------------------------------
# TRAIN MODEL
# ------------------------------------------------------
#results = model.train_model(0.2,"R8")
results = model.train_model(0.2,"OH")
#results = model.train_model(0.2,"R52")

# ------------------------------------------------------
# RESULTS 
# ------------------------------------------------------
print("RESULTS : ")
print(results)


