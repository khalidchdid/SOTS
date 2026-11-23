import numpy as np

np.set_printoptions(threshold=np.inf, linewidth=200)

data = np.load("/workspace/kakhmouch/data/PEMS04.npz", allow_pickle=True)

with open("/workspace/kakhmouch/SOTS_IMTS/experiments/PEMS04/aperçu_dataset.txt", "w") as f:
    f.write(f"Clés disponibles : {list(data.keys())}\n")

    for key in data.keys():
        arr = data[key]
        f.write(f"\n{'='*60}\n")
        f.write(f"--- {key} ---\n")
        f.write(f"shape : {arr.shape}\n")
        f.write(f"dtype : {arr.dtype}\n")

        if arr.dtype == object:
            f.write("contenu (objet) :\n")
            f.write(str(arr) + "\n")
        else:
            if np.issubdtype(arr.dtype, np.number):
                f.write(f"min : {np.nanmin(arr) if arr.size else None}\n")
                f.write(f"max : {np.nanmax(arr) if arr.size else None}\n")
                f.write(f"mean : {np.nanmean(arr) if arr.size else None}\n")
                nb_nan = np.isnan(arr).sum() if np.issubdtype(arr.dtype, np.floating) else "N/A"
                f.write(f"nb NaN : {nb_nan}\n")
            f.write("contenu complet :\n")
            f.write(np.array2string(arr) + "\n")

print("Écrit dans apercu_dataset.txt")

