from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from modules.utilities.sql_utils import *
from modules.utilities.cmp_pdf import Cpdf_cmp
from modules.signatures.signature_computer import *
from modules.coreset_selection.main import import_points
                
def recursive_merge(inter, start_index = 0):
    for i in range(start_index, len(inter) - 1):
        if inter[i][1] + 0.2 >= inter[i+1][0] and inter[i][2] == inter[i+1][2]:
            new_start = inter[i][0]
            new_end = inter[i+1][1]
            inter[i] = [new_start, new_end, inter[i][2]]
            del inter[i+1]
            return recursive_merge(inter.copy(), start_index=i)
    return inter

def report_inventory_table(Database, data_dir):
    Database.Create_table_reports_inventory()

    for folder in Path(data_dir).iterdir():
        for sig in Path(folder).iterdir():
            name = sig.name
            tested_module = name[:4]
            str_campaign = name[5:8]
            str_instantiation_dir = name[-1]
            compressed_path = str(sig)
            for signal in sig.iterdir():
                        if signal.name[0] == "c":
                            str_signal_name = signal.name[4: -4]    
                            Database.insert_reports_inventory(tested_module=tested_module, campaign=str_campaign, signal_name=str_signal_name,
                                                        rule=str_instantiation_dir, equality_state=False, path=compressed_path,
                                                        print_bool = False)

def discrepancy_table(Database):
    Database.Create_table_discrepanies()
    reports_table = Database.get_table('reports_inventory').fetchall()

    for i, row in enumerate(reports_table):
            signal_id, tested_module, campaign, signal_name, rule, signal_type, equality_state, path = row

            petit_pdf_path = os.path.join(path, 'cmp_'+ signal_name + '.pdf')
            Obj_petit_pdf = Cpdf_cmp(petit_pdf_path)

            #___ Extract errors intervals and merge overlapped intervals ___
            list_tuples_yellow_rects = Obj_petit_pdf.get_yellow_time()
            # merge intervals by a recursive technique
            list_error_intervals_not_sorted_on_start = recursive_merge(list_tuples_yellow_rects.copy())
            # sort the list from the start_time
            list_error_intervals = sorted(list_error_intervals_not_sorted_on_start)
            # __________________________

            for error_interval in list_error_intervals:
                start_time, end_time, discrepancy_type = error_interval
                Database.insert_discrepancies(signal_id=signal_id, start_time=start_time, end_time=end_time, discrepancy_type=discrepancy_type)

def distribution_visual_pca2D(X, output_path):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c="steelblue", s=8, alpha=0.5)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title("PCA of signature points")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)

def distribution_visual_pca2D_colored(X, labels, output_path):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 7))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        mask = np.array(labels) == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                c=[color], s=10, alpha=0.6, label=str(label))

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA of signature points — colored by class")
    ax.legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=2)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)

def distribution_visual_pca3D(X, output_path):
    pca = PCA(n_components=3)
    X_3d = pca.fit_transform(X)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c="steelblue", s=8, alpha=0.5)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")
    ax.set_title("PCA 3D - signature points")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)

def distribution_visual_tsne2D_coreset(X, selected_idx, output_path):
    tsne = TSNE(n_components=2, perplexity=30, random_state=0)
    X_2d = tsne.fit_transform(X)
    C_2d = tsne.fit_transform(X[selected_idx])

    plt.figure(figsize=(8, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c="steelblue", s=8, alpha=0.4, label="Full dataset")
    plt.scatter(C_2d[:, 0], C_2d[:, 1], c="orange", s=30, label="Coreset", zorder=5)
    plt.title("t-SNE of signature points with selected points(greedy)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)



def distribution_visual_tsne2D_colored(X, labels, output_path):
    tsne = TSNE(n_components=2, perplexity=30, random_state=0)
    X_2d = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 7))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        mask = np.array(labels) == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                c=[color], s=10, alpha=0.6, label=str(label))

    ax.set_title("t-SNE of signature points — colored by class")
    ax.legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=2)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)

def distribution_visual_tsne3D(X, output_path):
    tsne = TSNE(n_components=3, perplexity=30, random_state=0)
    X_3d = tsne.fit_transform(X)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c="steelblue", s=8, alpha=0.5)
    ax.set_title("t-SNE 3D - signature points")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)

def distribution_visual_tsne3D_colored(X, labels, output_path):
    tsne = TSNE(n_components=3, perplexity=30, random_state=0)
    X_3d = tsne.fit_transform(X)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        mask = np.array(labels) == label
        ax.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2],
                c=[color], s=10, alpha=0.6, label=str(label))

    ax.set_title("t-SNE 3D - signature points colored by class")
    ax.legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=2)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)

def get_labels(Database, ids):
    cursor = Database.con.cursor()

    select_labels_in_dataset =f"""SELECT discrepancy_id, label
                        FROM discrepancies
                        WHERE discrepancy_id IN {ids}
                    """
    data = cursor.execute(select_labels_in_dataset)
    labels = [row[1] for row in data]
    return labels

def signature_computation():
    base = Path(__file__).parent
    data_dir = os.path.join(base, "data_one_point")
    database_path = os.path.join(base, "outputs")
    visuals_path = os.path.join(base, "visualization")
    database_name = "database_one_point_2.db"
    dataset_name = "dataset_one_point_2"
    output_path = os.path.join(database_path, database_name)
    Database = automatic_reports_analysis_db("v0", output_path)

    report_inventory_table(Database, data_dir)
    discrepancy_table(Database)
    print(Database.len_table('discrepancies'))

    raw_table = discrepancies_table(database_path, database_name, dataset_name, window_percentage=0.01, signal_type=0).discrepancies_table
    print(len(raw_table))
    table = Interpolator().process(raw_table)
    visuals_path_discrepancies = os.path.join(visuals_path, "discrepancies_table_one_point_2.csv")
    Interpolator().export(table, visuals_path_discrepancies)
    signature_table, indices = SignatureComputer().process(table)
    visuals_path_signatures = os.path.join(visuals_path, "signatures_table_one_point_2.csv")
    SignatureComputer().export(signature_table, indices, visuals_path_signatures)

if __name__ == "__main__":
     database_path = "/workspace/kakhmouch/data/database_analog_F1.db"
     Database = automatic_reports_analysis_db("v0", database_path)
     path_signatures = "/workspace/kakhmouch/data/analog/5/signatures.csv"
     output_path = "/workspace/kakhmouch/SOTS_IMTS/tests/coreset_selection/tsne_2D_coreset_weighted.png"
     selected_path = "/workspace/kakhmouch/data/analog/5/discrepancies_selected.txt"

     original_indices, points = import_points(path_signatures)
     #ids = tuple(original_indices)
     
     selected_discrepancies = set()

     with open(selected_path, "r") as f:
        for x in f.read().split():
            selected_discrepancies.add(int(x))
     
     selected_idx = []
     for i, _ in enumerate(points):
         if original_indices[i] in selected_discrepancies:
             selected_idx.append(i)

     #labels = get_labels(Database, ids)
     #distribution_visual_tsne2D_colored(points, labels, output_path)

     distribution_visual_tsne2D_coreset(points, selected_idx, output_path)