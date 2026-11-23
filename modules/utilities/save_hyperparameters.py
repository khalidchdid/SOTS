import os
from datetime import datetime
    

def save_hyperparameters(path: str, **kwargs) -> None:
    """
    Save hyperparameters to a .txt file.
    
    Usage:
        save_hyperparameters(
            "experiments/results/hyperparameters.txt",
            k=200, L=100, order=2, p=2, n_iter=500, lr=0.05
        )
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(f"Run date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 30 + "\n")
        for key, value in kwargs.items():
            f.write(f"{key} : {value}\n")
    print("Hyperparameters")
    print("-" * 30)
    for key, value in kwargs.items():
        print(f"{key} : {value}\n")
    print(f"Hyperparameters saved to {path}")