from modules.utilities.sql_utils import *
from modules.utilities.signal_sivat_oracle import CSignalSivatOracle
from modules.utilities.terminal_style import printProgressBar, Input_style, end_style

import sys
import os
import toml
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def normalize(liste1,liste2):
        vmin1 = np.min(liste1)
        vmin2 = np.min(liste2)
        vmax1 = np.max(liste1)
        vmax2 = np.max(liste2)

        vmin = min(vmin1, vmin2)
        vmax = max(vmax1, vmax2)
        # delta1 = vmax1-vmin1
        # delta2 = vmax2-vmin2
        
        delta = abs(vmax-vmin) + 1e-3
        new_list_1 = [(float(i)-vmin)/ delta for i in liste1]
        new_list_2 = [(float(i)-vmin)/ delta for i in liste2]

        return new_list_1, new_list_2

def getdataset(database, name):
    """
    Return : Joint table of discrepanc and discrepancies : discrepancies, signal_name, path rule...
    """
    cursor = database.con.cursor()
    # select all the dataset
    select_discrepancy_in_dataset =f"""SELECT DISTINCT discrepancy_id
                        FROM {name}
                        ORDER BY discrepancy_id
                    """
    data = cursor.execute(select_discrepancy_in_dataset)

    discrepancies_selected = tuple([row[0] for row in data])
    data = database.get_discrepancies_map(discrepancies_selected)
    return data

def generate_dataset(Database, data_path, dataset_name, dataset_path, window_percentage, max_time_serie_length=400):
    # get discrepancy in {dataset_name} infos : discrepancy_id, reports_inventory.signal_id, path, signal_name, start_time, end_time, label
    data = getdataset(database=Database, name=dataset_name)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    label_dataframe = pd.DataFrame(columns=['path', 'index', 'label'])
    path_label =  os.path.join(dataset_path, 'labels.csv')
    for i, row in enumerate(data):
        ui_iteration = f"Number of discrepancies : {str(i)} "
        # for time execution optimization we only store the csv each 100 iteration
        if i % 100 == 0:
            print(ui_iteration)
            # write_label = int(input('wrote label :') or 0)
            if os.path.exists(path_label):
                os.remove(path_label)
            label_dataframe.to_csv(path_label, index = False)

        discrepancy_id = row[0]
        path = row[2]
        path = path.replace("\\", "/")
        signal_name = row[3]
        start_time = row[4] # % percentage (0< value <100)
        end_time = row[5]   # % percentage (0< value <100)
        label = row[6]

        # sometimes start_time == end_time, this produces errors
        if start_time == end_time:
            start_time = start_time - 0.02
            end_time = end_time + 0.02
        
        path_rule = os.path.join(data_path, path)
        signal_obj = CSignalSivatOracle(path_rule, 'n_' + signal_name)
        oracle_shape = signal_obj.get_oracle_shape()
        sivat_shape = signal_obj.get_sivat_shape()

        # percentage is transformed to a vector index
        n0_error_oracle = max(0,round((start_time/100) * oracle_shape))            
        n1_error_oracle = min(round((end_time/100) * oracle_shape), oracle_shape-1)
        
        # we extract some time points before and after the error for a better contextualisation
        n0_error_range_oracle = max(0, round((start_time/100 - window_percentage) * oracle_shape))
        n1_error_range_oracle = min(round((end_time/100 + window_percentage) * oracle_shape), oracle_shape-1)
        n0_error_range_sivat = max(0, round((start_time/100 - window_percentage) * sivat_shape))
        n1_error_range_sivat = min(round((end_time/100 + window_percentage) * sivat_shape), sivat_shape-1)

        oracle_values_0       = signal_obj.get_oracle_values()[n0_error_range_oracle:n1_error_range_oracle]
        oracle_validities   = signal_obj.get_oracle_validities()[n0_error_range_oracle:n1_error_range_oracle]
        oracle_time         = signal_obj.get_oracle_time()[n0_error_range_oracle:n1_error_range_oracle]
        sivat_values_0        = signal_obj.get_sivat_values()[n0_error_range_sivat:n1_error_range_sivat]
        sivat_validities    = signal_obj.get_sivat_validities()[n0_error_range_sivat:n1_error_range_sivat]
        sivat_time          = signal_obj.get_sivat_time()[n0_error_range_sivat:n1_error_range_sivat]
        
        # We create a yellow rectangle time series with the same size as oracle
        error               = np.zeros_like(oracle_values_0)
        n0_error = n0_error_oracle - n0_error_range_oracle
        n1_error = n1_error_oracle - n0_error_range_oracle
        # print(n0_error, n1_error, n0_error_oracle, n1_error_oracle)
        pos_error = np.r_[n0_error:n1_error]
        error[pos_error] = 1
        
        oracle_values, sivat_values = normalize(oracle_values_0, sivat_values_0)

        # downsampling (see downsampling part in False positive detection in I&C testing)
        downsampling = 1
        while len(error)// downsampling > max_time_serie_length:
            downsampling += 1
        
        # now we write the values to a float16
        oracle_values       = np.array(np.array(oracle_values[::downsampling]    ), dtype = np.float16).tolist()
        sivat_values        = np.array(np.array(sivat_values[::downsampling]     ), dtype = np.float16).tolist()
        oracle_validities   = np.array(oracle_validities[::downsampling]).tolist()
        oracle_time         = np.array(np.array(oracle_time[::downsampling]     ) * 1e-3, dtype= np.float16).tolist()
        sivat_validities    = np.array(sivat_validities[::downsampling]).tolist()
        sivat_time          = np.array(np.array(sivat_time[::downsampling]       ) * 1e-3 , dtype= np.float16).tolist()
        error               = np.array(error[::downsampling]).tolist()

        # vector describing the modalities by an index : ex : sivat validities = [1, 0, 1, 1] sivat validities modalitie = [4, 4, 4, 4] if 4 were describing sivat validitis modality
        oracle_values_mod       = np.array(np.ones_like(oracle_values, dtype=np.int8)       * 0).tolist()
        oracle_validities_mod   = np.array(np.ones_like(oracle_validities, dtype=np.int8)   * 1).tolist()
        sivat_values_mod        = np.array(np.ones_like(sivat_values, dtype=np.int8)        * 2).tolist()
        sivat_validities_mod    = np.array(np.ones_like(sivat_validities, dtype=np.int8)    * 3).tolist()
        error_mod               = np.array(np.ones_like(error, dtype=np.int8)               * 4).tolist()

        # the first two are used as static parameter in the model
        # (we concatenate this with the final representation before classification)
        concatenation_5 = [(0, 5, start_time/100)]
        concatenation_6 = [(0, 6,(end_time-start_time)/100)]

        concatenation_0 = list(zip(oracle_time, oracle_values_mod, oracle_values))
        concatenation_1 = list(zip(oracle_time, oracle_validities_mod, oracle_validities))
        concatenation_2 = list(zip(sivat_time, sivat_values_mod, sivat_values))
        concatenation_3 = list(zip(sivat_time, sivat_validities_mod, sivat_validities))
        concatenation_4 = list(zip(oracle_time, error_mod, error))
        
        df5 = pd.DataFrame(concatenation_5, columns = ['Time','Parameter','Value'])
        df6 = pd.DataFrame(concatenation_6, columns = ['Time','Parameter','Value'])
        df0 = pd.DataFrame(concatenation_0, columns = ['Time','Parameter','Value'])
        df1 = pd.DataFrame(concatenation_1, columns = ['Time','Parameter','Value'])
        df2 = pd.DataFrame(concatenation_2, columns = ['Time','Parameter','Value'])
        df3 = pd.DataFrame(concatenation_3, columns = ['Time','Parameter','Value'])
        df4 = pd.DataFrame(concatenation_4, columns = ['Time','Parameter','Value'])

        result_df = pd.concat([df0, df1, df2, df3, df4, df5, df6], axis=0, ignore_index=True)
        result_df2 = result_df.sort_values(by=['Time', 'Parameter'], ignore_index = True)
        
        path_unit = os.path.join(dataset_path, str(discrepancy_id) + '.txt')
        if not os.path.exists(path_unit):
            result_df2.to_csv(path_unit, index = False)

        # There is no classe E005 in the dataset (so remember to replace it with 'to justify' when writing PDF)
        # You can also replace non existing classe with 0 here (e.x. if label == 5, ... = path, discrepancy_id, 0)
        if label is None:
            label_dataframe.loc[len(label_dataframe), label_dataframe.columns] = path, discrepancy_id, 0
        elif label == 5:
            label_dataframe.loc[len(label_dataframe), label_dataframe.columns] = path, discrepancy_id, 5
        else :
            label_dataframe.loc[len(label_dataframe), label_dataframe.columns] = path, discrepancy_id, label


    if os.path.exists(path_label):
        os.remove(path_label)
        label_dataframe.to_csv(path_label, index = False)

    print(ui_iteration)

    print("stored in : " + dataset_path)
    print("Finished !!")
