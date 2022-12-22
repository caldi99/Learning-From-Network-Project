from pathlib import Path

# ------------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------------
#BASE PATH
BASE_PATH = str(Path(__file__).parent.parent.parent.resolve())
#PATHS DATASETS
PATH_BASE_DATASET = BASE_PATH + "\\datasets\\"
PATH_R8_DATASET = PATH_BASE_DATASET + "r8\\"
PATH_OH_DATASET = PATH_BASE_DATASET + "oh\\"
PATH_R52_DATASET = PATH_BASE_DATASET + "r52\\"
#PATH glove.6B.50d.txt 
PATH_GLOVE_6B_50D = BASE_PATH + "\\glove_6B_50d\\glove.6B.50d.txt"

# ------------------------------------------------------------------------
# DATASETS INFOS
# ------------------------------------------------------------------------
NUMBER_CLASSES_R8 = 8
NUMBER_CLASSES_OH = 23
NUMBER_CLASSES_R52 = 52

CLASSES_R52_DATASET = ['earn', 'interest', 'acq', 'sugar', 'coffee', 'ship', 'money-fx', 'gas', 'copper', 'veg-oil', 'crude', 'nat-gas', 'fuel', 'livestock', 'grain', 'meal-feed', 'ipi', 'pet-chem', 'reserves', 'gold', 'trade', 'gnp', 'cpi', 'jobs', 'money-supply', 'tea', 'cocoa', 'alum', 'heat', 'rubber', 'cotton', 'strategic-metal', 'tin', 'wpi', 'instal-debt', 'zinc', 'nickel', 'bop', 'lead', 'dlr', 'potato', 'iron-steel', 'orange', 'retail', 'lei', 'lumber', 'carcass', 'income', 'platinum', 'jet', 'housing', 'cpu']
CLASSES_OH_DATASET = ['C04', 'C07', 'C21', 'C17', 'C13', 'C11', 'C14', 'C15', 'C08', 'C12', 'C23', 'C18', 'C20', 'C10', 'C05', 'C06', 'C02', 'C01', 'C16', 'C19', 'C09', 'C22', 'C03']
CLASSES_R8_DATASET = ['earn', 'acq', 'interest', 'ship', 'money-fx', 'crude', 'grain', 'trade']

# ------------------------------------------------------------------------
# GENERAL CONSTANTS
# ------------------------------------------------------------------------
#ENCODING
ENCODING_TYPE = "utf8"
#float32
FLOAT_32 = "float32"
#FIELD CSV
FIELD_CSV_TEXT = "text"
FIELD_CSV_INTENT = "intent"
#ACTIVATION FUNCTIONS
ACTIVATION_FUNCTION_RELU = "relu"
ACTIVATION_FUNCTION_SOFTMAX = "softmax"
#LOSS FUNCTIONS
LOSS_SPARSE_CATEGORICAL_CROSS_ENTROPY = "sparse_categorical_crossentropy"
#OPTIMIZERS
OPTIMIZER_ADAM = "adam"
#METRICS
ACCURACY_METRIC= "accuracy"
#MAXIMUM NUMBER WORDS
MAXIMUM_NUMBER_WORDS = 50


# ------------------------------------------------------------------------
# RNN PARAMETERS
# ------------------------------------------------------------------------
RNN_NUMBER_HIDDEN_LAYERS = 3
RNN_NUMBER_GRU_LAYERS = 256
RNN_NUMBER_DENSE_LAYERS = 256
RNN_RECURRENT_DROPOUT = 0.2
RNN_EPOCHS = 10
RNN_BATCH_SIZE = 128

# ------------------------------------------------------------------------
# BOOSTING PARAMETERS
# ------------------------------------------------------------------------
BOOSTING_NUMBER_ESTIMATORS = 100

# ------------------------------------------------------------------------
# GCNN PARAMETERS
# ------------------------------------------------------------------------
FILE_OPEN_MODALITY = "a"

#Paths
GCNN_PATH_DATASET_SPLIT = BASE_PATH + "\\code\\graph_convolutional_neural_network\\datasets\\"
GCNN_PATH_DUMP_OBJECTS = GCNN_PATH_DATASET_SPLIT + "dumpS\\"
GCNN_PATH_DATASET_CORPUS = GCNN_PATH_DATASET_SPLIT + "corpus\\"
GCNN_DATASET_FILE_FORMAT = ".txt"

GCNN_LANGUAGE_WORDS = "english"
GCNN_TYPE_WORDS = "stopwords"
GCNN_THRESHOLD_WORD_FREQUENCY = 5
GCNN_WORD_EMBEDDINGS_DIM = 50
GCNN_WINDOW_SIZE = 20

GCNN_EPHOCS = 10
GCNN_EARLY_STOPPING = 3
GCNN_LEARNING_RATE = 0.02