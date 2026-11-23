"""
Sub module for deploiment cycle :
Instead of extracting a txt file and then pass into the model,
we extract directly information from raw data and database, then passes it into the model
to use, see label_database.py
"""

import numpy as np
import os
from modules.utilities.signal_sivat_oracle import CSignalSivatOracle
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Dict
from modules.model.imts_transformer_dense import DenseIMTSTransformer
from torch import load, no_grad
from configparser import ConfigParser

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

def collate_dense(batch):
    feats, statics_5, statics_6, labels = zip(*batch)
    lengths = [f.shape[0] for f in feats]
    Lmax = max(lengths)
    M2 = feats[0].shape[1]          # 2*M

    pad_feat = torch.zeros(len(batch), Lmax, M2)
    pad_mask = torch.ones(len(batch), Lmax, dtype=torch.bool)

    for i, f in enumerate(feats):
        l = f.shape[0]
        pad_feat[i, :l, :] = f
        pad_mask[i, :l] = False

    return pad_feat, pad_mask, torch.tensor(statics_5).unsqueeze(1), torch.tensor(statics_6).unsqueeze(1), torch.tensor(labels)



class DenseDFDataset(nn.Module):
    """
    Each TXT file: rows "t, param,  valr"
    All times are int
    """
    def __init__(self,
                dataframe,
                static_5,
                static_6,
                n_modalities: int):
        self.dataframe = dataframe
        self.static_5 = static_5
        self.static_6 = static_6
        self.M = n_modalities

    
    def _load_one(self) -> Tuple[torch.Tensor, torch.Tensor]:
        with no_grad():
            rows = []
            # rows [(time, param, valu), (time, param, val)...]
            rows = list(zip(*map(self.dataframe.get, self.dataframe)))
            rows.sort(key=lambda x: x[0])
            times = torch.tensor([r[0] for r in rows], dtype=torch.float32)
            mods  = torch.tensor([r[1] for r in rows], dtype=torch.long)
            vals  = torch.tensor([r[2] for r in rows], dtype=torch.float32)

            uniq_times, inv = torch.unique(times, return_inverse=True)
            L = len(uniq_times)
            V = torch.zeros((L, self.M))
            M = torch.zeros_like(V)
            V[inv, mods] = vals
            M[inv, mods] = 1.0

            feat = torch.cat([V, M], dim=1)  # (L,2M)
            return feat, torch.tensor(self.static_5, dtype=torch.float32), torch.tensor(self.static_6, dtype=torch.float32),torch.tensor(0, dtype=torch.long)
    
    def __getitem__(self, i): return self._load_one()
    def __len__(self): return 1


def unit_loader(execution_round_path, path_rule1, signal_type, signal_name, start_time, end_time, window_percentage, max_time_serie_length = 400):
    # sometimes start_time == end_time, this produces errors
    if start_time == end_time:
        start_time = start_time - 0.02
        end_time = end_time + 0.02

    path_rule = os.path.join(execution_round_path, path_rule1)
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
    error               = np.zeros_like(oracle_values_0, dtype=bool)
    n0_error = n0_error_oracle - n0_error_range_oracle
    n1_error = n1_error_oracle - n0_error_range_oracle
    # print(n0_error, n1_error, n0_error_oracle, n1_error_oracle)
    pos_error = np.r_[n0_error:n1_error]
    error[pos_error] = 1

    if signal_type == True:
        # see normalization in Internship report
        oracle_values, sivat_values = normalize(oracle_values_0, sivat_values_0)
    else:
        oracle_values, sivat_values = oracle_values_0, sivat_values_0

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
    static_5, static_6 = start_time/100, (end_time-start_time)/100
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
    return result_df2, static_5, static_6

def get_hyperparam(CONFIG_PATH):
    """
    return : d_model, n_heads, n_layers, n_classes, n_modalities
    """
    config = ConfigParser()
    config.read(CONFIG_PATH)
    d_model         = int(config['main']['d_model'])
    n_heads         = int(config['main']['n_heads'])
    n_layers        = int(config['main']['n_layers'])
    n_classes       = int(config['main']['n_classes'])
    n_modalities    = int(config['main']['n_modalities'])

    return d_model, n_heads, n_layers, n_classes, n_modalities

if __name__ == "__main__":

    execution_round_path = r'C:/Users/aelhaoufi/Documents/data_set_rapport/2 - Execution rounds'
    MODELS_path = 'C:/Users/aelhaoufi/Automatic_Test_report_Analysis_v2/Transformer/'

    model_id_str = '37'
    MODEL_PATH = os.path.join(MODELS_path, model_id_str, 'best_auroc_'+ model_id_str +'.pt')
    CONFIG_PATH = os.path.join(MODELS_path, model_id_str, 'metadata_' + model_id_str + '.txt')  
    
    d_model, n_heads, n_layers, n_classes, n_modalities = get_hyperparam(CONFIG_PATH=CONFIG_PATH)

    batch_size   = 1
    device       = 'cuda'

    path_rule1 = 'SI07/SI07_TC4/output/results/1'
    signal_type = 1
    signal_name = 'SI07_D2_E23_A21'
    start_time, end_time = 6.32809, 8.388615
    window_percentage = 0.005 # It should be the same as the dataset (default 0.005)


    model = DenseIMTSTransformer(n_modalities=n_modalities,
                        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                        n_classes=n_classes).to(device)
    # import a model to deploy
    model.load_state_dict(load(MODEL_PATH, weights_only=True))
    
    dataframe, static_5, static_6 = unit_loader(execution_round_path, path_rule1, signal_type, signal_name, start_time, end_time, window_percentage)
    dataset = DenseDFDataset(dataframe=dataframe, static_5 = static_5, static_6 = static_6, n_modalities=n_modalities)
    loader  = DataLoader(dataset, batch_size=1, collate_fn=collate_dense)

    signal_path = os.path.join(execution_round_path, path_rule1, 'cmp_'+ signal_name + '.pdf')
    
    y_true, y_pred, y_prob = [], [], []
    with no_grad():
        for feat, pad_mask, static_5, static_6, y in loader:
            feat, static_5, static_6, pad_mask = feat.to(device), static_5.to(device), static_6.to(device), pad_mask.to(device)
            logits = model(feat, pad_mask, static_5, static_6)
            preds = logits.argmax(dim=1)
            probs = logits.softmax(dim=1)
            print(y)
            print(y.shape, logits.shape)

            print("logits")
            print(logits)

            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
            print("preds")
            print(y_pred)
            print("probs")
            print(y_prob)
