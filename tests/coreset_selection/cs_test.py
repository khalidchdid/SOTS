import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def plot_sw_from_log(log_path: str, save_path: str = None) -> None:
    """
    Plot SW distance evolution from a greedy selection log file.

    Args:
        log_path  : path to the .txt log file (CSV format with columns
                    step, candidate_idx, sw_distance)
        save_path : if provided, save the figure instead of displaying it
    """

    df = pd.read_csv(log_path, sep=',')

    plt.figure(figsize=(8, 5))
    plt.plot(df['step'], df[' sw_distance'], marker='o', markersize=3, linewidth=1)
    plt.xlabel("Coreset size (k)")
    plt.ylabel("Sliced Wasserstein distance")
    plt.title("Evolution of SW distance during greedy coreset selection")
    plt.grid(True, alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    #log_path = "/workspace/kakhmouch/data/analog/4/sw_steps.txt"
    #save_path = "/workspace/kakhmouch/SOTS_IMTS/tests/coreset_selection/greedy_cs_2.png"
    #plot_sw_from_log(log_path=log_path, save_path=save_path)

    # ── Fill in your values here ──────────────────────────────────
    class_names = [
        "Class 1", "Class 2", "Class 3", "Class 4",
        "Class 5", "Class 6", "Class 7", "Class 8", "Class 9", "Class 10", "Class 11"
    ]

    # Number of examples per class for each configuration
    counts_full    = [453, 1652, 2788, 377, 0, 118, 3088, 144, 0, 64, 87]  # full dataset
    counts_greedy  = [42, 83, 203, 31, 0, 18, 283, 17, 0, 4, 19]  # coreset greedy
    counts_weighted= [31, 149, 246, 59, 0, 32, 150, 17, 0, 12, 4]  # coreset weighted
    counts_random  = [35, 104, 206, 46, 0, 15, 267, 6, 0, 1, 20]  # random sampling
    # ─────────────────────────────────────────────────────────────

    n_classes = len(class_names)
    x = np.arange(n_classes)
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))

    bars1 = ax.bar(x - 1.5*width, counts_full,     width, label="Full dataset",      color="steelblue",  alpha=0.85)
    bars2 = ax.bar(x - 0.5*width, counts_greedy,   width, label="Coreset Greedy",    color="orange",     alpha=0.85)
    bars3 = ax.bar(x + 0.5*width, counts_weighted, width, label="Coreset Weighted",  color="mediumseagreen", alpha=0.85)
    bars4 = ax.bar(x + 1.5*width, counts_random,   width, label="Random Sampling",   color="tomato",     alpha=0.85)

    ax.set_xlabel("Class")
    ax.set_ylabel("Number of examples")
    ax.set_title("Class distribution: Full dataset vs Coresets vs Random")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = "/workspace/kakhmouch/SOTS_IMTS/tests/coreset_selection/histogram_classes.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {save_path}")