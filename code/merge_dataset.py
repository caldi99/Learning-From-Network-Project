from utils.helper import Helper
from config import configs

helper = Helper()

print("MERGING R8 DATASET ..")
dataframes = helper.merge_dataset(configs.PATH_R8_DATASET)
helper.save_dataframe_to_csv(dataframes,configs.PATH_R8_DATASET + "complete.csv")

print("MERGING OH DATASET ..")
dataframes = helper.merge_dataset(configs.PATH_OH_DATASET)
helper.save_dataframe_to_csv(dataframes,configs.PATH_OH_DATASET + "complete.csv")

print("MERGING R52 DATASET ..")
dataframes = helper.merge_dataset(configs.PATH_R52_DATASET)
helper.save_dataframe_to_csv(dataframes,configs.PATH_R52_DATASET + "complete.csv")