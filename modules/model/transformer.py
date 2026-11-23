"""
Training a model 
PROBLEM  (1):
IN VALIDATION SET, we should find all different classes
Otherwise, we will have a bug
TO DO (1):
Time Series nbr Points
Normalization
# GRID SEARCH
infered from less represented classes
"""

import argparse
from pathlib import Path
import random
import os
from configparser import ConfigParser
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from modules.model.evaluation_metrics import evaluate

from time import time

from modules.model.warmup_module import WarmUpScheduler

from modules.model.imts_transformer_dense import (
    DenseIMTSTransformer,
    DenseTxtDataset,
    collate_dense
)


def transformer(id, model_path, dataset_train_path, dataset_val_path, d_model, n_heads, n_layers, n_classes, n_modalities, epochs, batch_size, lr, device):
    print("* CUDA available ===> ", torch.cuda.is_available())

    root_train = dataset_train_path
    root_val = dataset_val_path
    torch.manual_seed(8422)

    model_dir = model_path
    os.makedirs(model_dir, exist_ok=True)
    path_stat = os.path.join(model_dir, str(id) + ".csv")
    path_metadata = os.path.join(model_dir, "metadata_" + str(id) + ".txt")
    if os.path.exists(path_metadata):
        os.remove(path_metadata)
    
    stats_df = pd.DataFrame(columns=['Epoch', 'duration', 'TrainLoss', 'ValLoss', 'ValAcc', 'AUPRC', 'F1_macro', 'F1_per_class', 'conf_matrix', 'learning_rate'])
    config = ConfigParser()

    config.read(path_metadata)
    config.add_section('main')
    config.set('main', 'd_model',   str(d_model))
    config.set('main', 'n_heads',   str(n_heads))
    config.set('main', 'n_layers',  str(n_layers))
    config.set('main', 'n_classes', str(n_classes))
    config.set('main', 'batch_size',str(batch_size))
    config.set('main', 'n_modalities', str(n_modalities))

    # ---------------- Data ----------------
    # Expect labels.csv with columns: index,label
    labels_df_train = pd.read_csv(os.path.join(root_train, "labels.csv"))
    label_map_train = dict(zip(labels_df_train["index"], labels_df_train["label"]))

    labels_df_val = pd.read_csv(os.path.join(root_val, "labels.csv"))
    label_map_val = dict(zip(labels_df_val["index"], labels_df_val["label"]))

    all_indices_train = list(label_map_train.keys())
    all_indices_val = list(label_map_val.keys())
    # Select random from training, put on validation
    random.shuffle(all_indices_train)
    random.shuffle(all_indices_val)

    class_indices = {i: [key for key, label in label_map_train.items() if label == i] for i in range(n_classes)}

    total_count = len(all_indices_train)
    weights = []
    for i in range(n_classes):
        class_count = len(class_indices[i])
        # To avoid division by zero
        weights.append(total_count / class_count if class_count > 0 else 0)
        # write in config
        config.set('main', 'class_'+ str(i) + '_count', str(class_count))


    weights = torch.tensor(weights)
    
    train_idx = all_indices_train

    val_idx = all_indices_val

    random.shuffle(train_idx)
    random.shuffle(val_idx)

    print("* Validation set size: ", len(val_idx))
    print("* training set size: ", len(train_idx))
    print("* Number of classes: ", n_classes)

    train_ds = DenseTxtDataset(root_train, train_idx, label_map_train, n_modalities)
    val_ds   = DenseTxtDataset(root_val, val_idx, label_map_val, n_modalities)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_dense)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_dense)
    
    # ---------------- Model ----------------

    model = DenseIMTSTransformer(
        n_modalities=n_modalities,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        n_classes=n_classes).to(device)
    
    # --- Finetuning : if you want to retrain a model -----
    # Finetuning = 'finet_'
    # PATH = "./new_weights/conf25_best_acc_batch_size_"+ str(args.batch_size) +"_d_model_"+ str(d_model) +"_n_heads_"+ str(n_heads) +"_n_layers_"+ str(n_layers) + "_n_epoch_40" + ".pt"
    # model.load_state_dict(torch.load(PATH, weights_only=True))


    # hyperparams are weight_decay, factor, patience, cooldown
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max', factor=0.5,patience=3, cooldown=2)


    scheduler = WarmUpScheduler(optimizer, lr_scheduler=lr_scheduler, initial_lr=lr,
                                   len_loader=len(train_loader),
                                   warmup_steps=500,
                                   warmup_start_lr=0.0,
                                   warmup_mode='linear')

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights).to(device)
    best_auprc = 0.0
    best_f1_macro = 0.0
    best_acc = 0.0
    val_auprc = 0.0

    print("<_____________________________>")
    print("<____________Start____________>")

    config.set('main', 'train_size' , str(len(train_idx)))
    config.set('main', 'val_size'   , str(len(val_idx)))
    config.set('main', 'best_auprc' , str(best_auprc))
    config.set('main', 'best_f1_macro' , str(best_f1_macro))
    config.set('main', 'best_acc'   , str(best_acc))

    with open(path_metadata, 'w') as f:
        config.write(f)

    train_loss = 99
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        current_learning_rate = scheduler.get_last_lr()
        
        t0 = time()
        for feat, pad_mask, static_5, static_6, y in train_loader:
            feat, static_5, static_6, pad_mask = feat.to(device), static_5.to(device), static_6.to(device), pad_mask.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(feat, pad_mask, static_5, static_6)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.size(0)
            scheduler.step(-train_loss)
        dur = time() - t0

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc, f1_macro, f1_per_class, val_auprc, conf_matrix = evaluate(model, val_loader, device, n_classes=n_classes, criterion=criterion)
        if val_auprc > best_auprc:
            best_auprc = val_auprc
            config.set('main', 'best_auprc', str(best_auprc))
            with open(path_metadata, 'w') as f:
                config.write(f)
            dir = os.path.join(model_dir, "best_auprc_"+ str(id) +".pt")
            torch.save(model.state_dict(), dir)
        if f1_macro > best_f1_macro:
            best_f1_macro = f1_macro
            config.set('main', 'best_f1_macro', str(best_f1_macro))
            with open(path_metadata, 'w') as f:
                config.write(f)
            dir = os.path.join(model_dir, "best_f1_macro_"+ str(id) +".pt")
            torch.save(model.state_dict(), dir)
        if val_acc > best_acc:
            best_acc = val_acc
            config.set('main', 'best_acc', str(best_acc))
            with open(path_metadata, 'w') as f:
                config.write(f)
            dir = os.path.join(model_dir, "best_acc_"+ str(id) +".pt")
            torch.save(model.state_dict(), dir)
        stats_df.loc[len(stats_df), stats_df.columns] = epoch, dur, train_loss, val_loss, val_acc, val_auprc, f1_macro, f1_per_class, conf_matrix, current_learning_rate[0]
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs} | "
                f"duration={dur:.3f} | "
                f"TrainLoss={train_loss:.4f} | "
                f"ValLoss={val_loss:.4f} | "
                f"ValAcc={val_acc:.3f} | "
                f"AUPRC={val_auprc:.3f} | "
                f"F1 Macro={f1_macro*100:.2f}% | "
                f"learning_rate={current_learning_rate[0]:e}")
            print("  F1 per class :")
            for i, f1 in enumerate(f1_per_class):
                print(  f"    Class {i:2d} : {f1*100:.2f}%")
            print("-" * 50)
            print("  Confusion matrix :")
            print(conf_matrix)
            print("=" * 50)

        if os.path.exists(path_stat):
            os.remove(path_stat)
        stats_df.to_csv(path_stat, index = False)

    print(f"Training complete. Best validation AUPRC: {best_auprc:.3f}")
    print(f"Best validation F1 macro: {best_f1_macro:.3f}")
    print(f"Best validation ACC: {best_acc:.3f}")
    print("Saved model to best_model_{ i }_conf... for {0} auprc, {1} f1_macro, {2} acc")