

from config import configs
from rocchio.rocchio_model import RocchioModel

# ------------------------------------------------------
# CREATE MODEL
# ------------------------------------------------------
model = RocchioModel()

# ------------------------------------------------------
# TRAIN MODEL
# ------------------------------------------------------
results = model.train_model(0.2,"R8")
#results = model.train_model(0.2,"OH")
#results = model.train_model(0.2,"R52")

# ------------------------------------------------------
# RESULTS 
# ------------------------------------------------------
print("RESULTS : ")
print(results)


