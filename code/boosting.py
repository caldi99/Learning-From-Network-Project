

from config import configs
from boosting.boosting_model import BoostingModel

# ------------------------------------------------------
# CREATE MODEL
# ------------------------------------------------------
model = BoostingModel()

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


