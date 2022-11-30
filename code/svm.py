

from config import configs
from svm.svm_model import SVMModel

# ------------------------------------------------------
# CREATE MODEL
# ------------------------------------------------------
model = SVMModel()

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


