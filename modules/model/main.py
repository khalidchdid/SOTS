import numpy as np

from modules.utilities.sql_utils import *
from SOTS_IMTS.modules.model.transformer import transformer

def main(id):
    dataset_train_path = "../data/dataset_binary_sots/"
    dataset_val_path = "../data/dataset_binary_sots_val/"
    model_path = f"../data/models/binary/{id}"

    # Running the model
    transformer(
        id=id, 
        model_path=model_path, 
        dataset_train_path=dataset_train_path,
        dataset_val_path=dataset_val_path,
        d_model=64,
        n_heads=4, 
        n_layers=2, 
        n_classes=12, 
        n_modalities=7, 
        epochs=150, 
        batch_size=64,
        lr=1e-3, 
        device="cuda")
    # put the rest of the dataset for validation and train on the whole data -> modify transformer



if __name__ == "__main__":
    main(id=1)