from config import configs
from bagging.bagging_model import BaggingModel

# ------------------------------------------------------
# CREATE MODEL
# ------------------------------------------------------
model = BaggingModel()

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


