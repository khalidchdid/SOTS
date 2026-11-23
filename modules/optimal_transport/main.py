import torch
import numpy as np

from SOTS_IMTS.modules.optimal_transport.sliced_wasserstein import SlicedWasserstein
from SOTS_IMTS.modules.optimal_transport.torch_sliced_wasserstein import SlicedWassersteinTorch

def main(n: int):
    if n == 1:
        X = np.array(([0, 0, 0], [0.1, 0.1, 0.1], [0.9, 0.9, 0.9], [1, 1, 1], [2, 2, 2], [2.2, 2.01, 2.01], [4, 4, 4], [4.3, 4, 3.99]))
        sw = SlicedWasserstein(X, p=2)
        C1 = np.array(([0, 0, 0], [1, 1, 1], [2, 2, 2], [4, 4, 4]))
        C2 = np.array(([0, 0, 0], [4 ,4, 4], [3, 3, 3]))
        C3 = np.array(([1, 1, 1], [0.9, 0.9, 0.9], [0.1, 0.1, 0.1], [2, 2, 2], [2.2, 2.01, 2.01]))
        d1 = sw.compute(C1)
        d2 = sw.compute(C2)
        d3 = sw.compute(C3)
        print(d1)
        print(d2)
        print(d3)

    elif n == 2:
        u_np = np.array([[0, 0, 0], [0.1, 0.1, 0.1], [0.9, 0.9, 0.9], [1, 1, 1], [2, 2, 2], [2.2, 2.01, 2.01], [4, 4, 4], [4.3, 4, 3.99]])
        v_np = np.array([[[0, 0, 0], [1, 1, 1], [2, 2, 2], [4, 4, 4]]])

        u_np_ = np.array([[0, 0, 0], [0.1, 0.1, 0.1], [0.9, 0.9, 0.9], [1, 1, 1], [2, 2, 2], [2.2, 2.01, 2.01], [4, 4, 4], [4.3, 4, 3.99]])
        v_np_ = np.array([[1, 1, 1], [0.9, 0.9, 0.9], [0.1, 0.1, 0.1], [2, 2, 2], [2.2, 2.01, 2.01]])

        u_torch = torch.tensor(u_np, dtype=torch.float32)
        v_torch = torch.tensor(v_np, dtype=torch.float32)

        swn = SlicedWasserstein(u_np_)
        swt = SlicedWassersteinTorch(u_torch)
        result_torch = swt.compute_batch(v_torch)
        result_numpy = swn.compute(v_np_)

        print(f"torch : {result_torch}, \nnumpy : {result_numpy}")

if __name__ == "__main__":
    n = int(input("Enter the test number : "))
    main(n)