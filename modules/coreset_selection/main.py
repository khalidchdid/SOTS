from modules.data_preprocessing.discrepancies_table import discrepancies_table
from modules.data_preprocessing.interpolation import Interpolator
from modules.signatures.signature_computer import *
from modules.coreset_selection.greedy_cs import *
from modules.utilities.generate_dataset import generate_dataset

def import_points(input_path: str = "signatures.csv"):
    ids = []
    rows = []

    with open(input_path) as f:
        lines = f.readlines()

    for line in lines[1:]:
        parts = line.strip().split(";")
        ids.append(int(parts[0]))
        rows.append([float(x) for x in parts[1:]])

    points = np.array(rows)
    return ids, points

def main():
    input_path = "../data/signatures_binary.csv"
    data_path = "../data/"
    database_path = "../data/database_binary_selected.db"
    validation_path = "../data/database_binary_validation.db"
    discrepancies_selected_path = "../data/discrepancies_binary_selected.txt"
    discrepancies_path = "../data/discrepancies_binary.txt"
    old_dataset_name = "dataset_analog_F1"
    dataset_name = "dataset_binary_sots"
    dataset_val_name = "dataset_binary_sots_val"

    # Hyperparameters
    k = 700
    L = 100
    method = "exact"
    batch_size = 500

    # Points import
    original_indices, points = import_points(input_path)
    print()
    print(f"{len(points)} Points imported successfully")
    print()

    # Coreset Selection
    selector = GreedyCoresetSelector(points, p=2)
    selected_idx, history = selector.select(k=k, L=L, method=method, batch_size=batch_size)
    original_ids_selected = [original_indices[i] for i in selected_idx]

    print(f"\nFinal coreset size : {len(selected_idx)}")
    print(f"Final SW distance  : {history[-1]:.6f}")
    print()

    with open(discrepancies_selected_path, "w") as f:
        f.write(" ".join(map(str, original_ids_selected)))
    
    print("Selected discrepancies saved!")

    selector.plot_history(history)

    # Updating dataset
    Database = automatic_reports_analysis_db("v0", database_path)
    Database.delete_table("dataset_analog_F1")

    Database.create_table_dataset(dataset_name)

    with open(discrepancies_selected_path, "r") as f:
        for x in f.read().split():
            Database.insert_time_series(dataset_name, int(x))
    
    print(f"Training database updated successfully !")
    print(f"{Database.len_table(dataset_name)} training samples")

    # Generating training dataset
    window_percentage = 0.01
    max_time_serie_length = 400
    generate_dataset(Database, data_path, dataset_name, window_percentage, max_time_serie_length=max_time_serie_length)

    print(f"Training dataset generated successfully !")

    # Generating validation dataset
    DatabaseV = automatic_reports_analysis_db("v0", validation_path)
    DatabaseV.delete_table("dataset_analog_F1")

    DatabaseV.create_table_dataset(dataset_val_name)
    with open(discrepancies_path, "r") as f:
        for x in f.read().split():
            DatabaseV.insert_time_series(dataset_val_name, int(x))
    with open(discrepancies_selected_path, "r") as f:
        for x in f.read().split():
            DatabaseV.delete_line_from_table(dataset_val_name, int(x))

    print(f"Validation database updated successfully !")
    print(f"{DatabaseV.len_table(dataset_val_name)} validation samples")

    window_percentage = 0.01
    max_time_serie_length = 400
    generate_dataset(DatabaseV, data_path, dataset_val_name, window_percentage, max_time_serie_length=max_time_serie_length)

    print(f"Validation dataset generated successfully !")

if __name__ == "__main__":
    main()