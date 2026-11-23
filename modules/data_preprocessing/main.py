from modules.data_preprocessing.discrepancies_table import discrepancies_table
from modules.data_preprocessing.interpolation import Interpolator

import os


def main():
    data_path = "../data/"
    database_name = "database_analog_F1.db"
    dataset_name = "dataset_analog_F1"
    output_path = "../data/aligned_binary_discrepancies.csv"
    discrepancies_path = "../data/discrepancies_binary.txt"
    window_percentage = 0.01

    # Discrepancies_table
    print()
    table = discrepancies_table(data_path=data_path, database_name=database_name, dataset_name=dataset_name, window_percentage=window_percentage, signal_type=0).discrepancies_table
    discrepancies = [discrepancy[0] for discrepancy in table]
    with open(discrepancies_path, "w") as f:
        f.write(" ".join(map(str, discrepancies)))
    print(f"Loaded {len(table)} discrepancies")
    print()

    # Interpolation
    print()
    preprocessor = Interpolator()
    aligned = preprocessor.process(table)
    print(f"Aligned {len(aligned)} discrepancies (empty signals discarded)")
    print()
    
    preprocessor.export(aligned=aligned, output_path=output_path)


if __name__ == "__main__":
    main()
