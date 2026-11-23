""""
class representing the table of all discrepancies
discrepancy : (oracle_time, oracle_values, oracle_validities, sivat_time, sivat_values, sivat_validities)

"""

from modules.utilities.sql_utils import *
from modules.utilities.terminal_style import *
from modules.utilities.signal_sivat_oracle import *
from modules.data_preprocessing.interpolation import *

import os

class discrepancies_table():
    def __init__(self, data_path, database_name, dataset_name, window_percentage, signal_type = None):
        self.data_path = data_path
        self.database_name = database_name
        self.dataset_name = dataset_name
        self.window_percentage = window_percentage
        self.signal_type = signal_type
        self.get_database()
        self.get_discrepancies_table()
    
    def get_database(self):
        path = os.path.join(self.data_path, self.database_name)
        self.database = automatic_reports_analysis_db("v0", path)
    
    def get_dataset(self):
        """
        gets all the binary or analog discrepancies from the database
        """

        cursor = self.database.con.cursor()

        select_discrepancy_in_dataset =f"""SELECT d.discrepancy_id
                            FROM discrepancies d
                            WHERE EXISTS (
                                SELECT 1 FROM dataset_sots ds WHERE ds.discrepancy_id = d.discrepancy_id
                            )
                        """
        data = cursor.execute(select_discrepancy_in_dataset)

        discrepancies_selected = tuple([row[0] for row in data])
        data = self.database.get_discrepancies_map(discrepancies_selected)
        data_size = len(discrepancies_selected)
        return data, data_size
    
    def normalize(self, liste1, liste2):
        vmin1 = np.min(liste1)
        vmin2 = np.min(liste2)
        vmax1 = np.max(liste1)
        vmax2 = np.max(liste2)

        vmin = min(vmin1, vmin2)
        vmax = max(vmax1, vmax2)
        
        delta = abs(vmax-vmin) + 1e-3
        new_list_1 = [(float(i)-vmin)/ delta for i in liste1]
        new_list_2 = [(float(i)-vmin)/ delta for i in liste2]

        return new_list_1, new_list_2
    
    def get_discrepancies_table(self):
        discrepancies = []
        data, data_size = self.get_dataset()
        for i, row in enumerate(data):
            printProgressBar(i+1, data_size, prefix='Progress', suffix='Processed_discrepancies', length=50)
            id = row[0]
            path = row[2]
            path = path.replace("\\", "/")
            signal_name = row[3]
            start_time = row[4] # % percentage (0< value <100)
            end_time = row[5]   # % percentage (0< value <100)


            # sometimes start_time == end_time, this produces errors
            if start_time == end_time:
                start_time = start_time - 0.02
                end_time = end_time + 0.02

            path_rule = os.path.join(self.data_path, path)
            signal_obj = CSignalSivatOracle(path_rule, 'n_' + signal_name)
            oracle_shape = signal_obj.get_oracle_shape()
            sivat_shape = signal_obj.get_sivat_shape()

            # we extract some time points before and after the error for a better contextualisation
            n0_error_range_oracle = max(0, round((start_time/100 - self.window_percentage) * oracle_shape))
            n1_error_range_oracle = min(round((end_time/100 + self.window_percentage) * oracle_shape), oracle_shape-1)
            n0_error_range_sivat = max(0, round((start_time/100 - self.window_percentage) * sivat_shape))
            n1_error_range_sivat = min(round((end_time/100 + self.window_percentage) * sivat_shape), sivat_shape-1)

            oracle_values       = signal_obj.get_oracle_values()[n0_error_range_oracle:n1_error_range_oracle]
            oracle_validities   = signal_obj.get_oracle_validities()[n0_error_range_oracle:n1_error_range_oracle]
            oracle_time         = signal_obj.get_oracle_time_normalized()[n0_error_range_oracle:n1_error_range_oracle]
            sivat_values        = signal_obj.get_sivat_values()[n0_error_range_sivat:n1_error_range_sivat]
            sivat_validities    = signal_obj.get_sivat_validities()[n0_error_range_sivat:n1_error_range_sivat]
            sivat_time          = signal_obj.get_sivat_time_normalized()[n0_error_range_sivat:n1_error_range_sivat]

            oracle_values, sivat_values = self.normalize(oracle_values, sivat_values)

            discrepancy = (id, oracle_time, oracle_values, oracle_validities, sivat_time, sivat_values, sivat_validities)
            discrepancies.append(discrepancy)
        
        self.discrepancies_table = discrepancies