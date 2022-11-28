import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split


class Helper:
    def merge_dataset(self,path_list_csv_to_merge):
        path_files = self.get_path_csv_files(path_list_csv_to_merge)

        list_dataframes_to_merge = []

        #get dataframes to merge
        for path_file in path_files:
            list_dataframes_to_merge.append(pd.read_csv(path_file))
        
        #merge dataframes
        return pd.concat(list_dataframes_to_merge,ignore_index=True)

    def save_dataframe_to_csv(self, dataframe, path_to_write_csv):
        #save dataframe to correct path
        dataframe.to_csv(Path(path_to_write_csv),index=False)


    def get_path_csv_files(self,path_list_csv):
        # list paths of files to return
        list_paths = []

        #list of names of the files
        list_name_files = os.listdir(path_list_csv)
        
        for file_name in list_name_files:
            list_paths.append(os.path.join(path_list_csv,file_name))
        
        return list_paths
    
    def split_dataset(self, path_dataset_to_split, test_size):
        
        #read dataframe
        dataframe = pd.read_csv(path_dataset_to_split)

        #return splitted datframe
        return train_test_split(dataframe, test_size = test_size)


