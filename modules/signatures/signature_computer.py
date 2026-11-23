"""
    Computes truncated path signatures up to a given order for a discrepancy
    defined by two paths (values and validity).

    Each path includes time as its first component (time augmentation),
    so each path lives in R^3 : (time, signal_1, signal_2).

    The output is a tuple of two signature vectors (one per path) that
    together form the point representation of the discrepancy used in
    the OT coreset selection step.

    Args:
        order : truncation order (default: 2)

    Example:
        computer = SignatureComputer(order=2)

"""

import numpy as np
from typing import List, Tuple
import pandas as pd

from modules.data_preprocessing.discrepancies_table import *
from modules.data_preprocessing.interpolation import *


class SignatureComputer:
    def __init__(self, order: int = 2):
        self.order = order
    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    @staticmethod
    def _increments(path: np.ndarray) -> np.ndarray:
        """
        Compute the increments of a path.

        dx_t = x_{t+1} - x_t for each consecutive pair of points.

        Returns:
            np.array of shape (T-1, d)
        """
        return np.diff(path, axis=0)

    @staticmethod
    def _order1(dx: np.ndarray) -> np.ndarray:
        """
        Compute the order-1 signature terms.

        S^i = sum_t dx^i_t
        Total displacement of each component, equal to x^i_{T-1} - x^i_0.

        Args:
            dx : np.array of shape (T-1, d)
        Returns:
            np.array of shape (d,)
        """
        return dx.sum(axis=0)

    @staticmethod
    def _order2(dx: np.ndarray, rescaling: bool = False) -> np.ndarray:
        """
        Compute the order-2 signature terms.

        S^{ij} = sum_{s < t} dx^i_s * dx^j_t

        Single pass over increments: at each step t, accumulate
        outer(cumsum[:t], dx[t]), where cumsum[:t] is the sum of all
        increments strictly before t.
        Cumsum is updated AFTER the outer product to enforce s < t strictly.

        The (i,j) entry encodes the interaction between component i (past)
        and component j (present). In particular, S^{12} - S^{21} is the
        Levy area between components 1 and 2, capturing which signal moves
        first.

        Args:
            dx        : np.array of shape (T-1, d)
            rescaling : if True, multiply by 2! = 2 to compensate for the
                        natural decay of order-2 iterated integrals by O(1/2!)
        Returns:
            np.array of shape (d*d,) — flattened d x d matrix
        """
        d = dx.shape[1]
        order2 = np.zeros((d, d))
        cumsum = np.zeros(d)
        for t in range(len(dx)):
            order2 += np.outer(cumsum, dx[t])
            cumsum += dx[t]
        if rescaling:
            order2 *= 2
        return order2.flatten()

    def _compute_signature(self, path: np.ndarray) -> np.ndarray:
        """
        Compute the truncated signature of a single path up to self.order.
        Includes the order-0 term (= 1.0) by convention.

        For d=3 (time-augmented path) and order=2:
            output shape = (1 + 3 + 9,) = (13,)

        Args:
            path : np.array of shape (T, d)
        Returns:
            np.array of shape (1 + d + d*d,) for order=2
                              (1 + d,)        for order=1
        """
        dx = self._increments(path)
        terms = [np.array([1.0])]       # order 0
        terms.append(self._order1(dx))  # order 1 : shape (d,)
        if self.order >= 2:
            terms.append(self._order2(dx, rescaling=True))  # order 2 : shape (d*d,)
        return np.concatenate(terms)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def compute(self, values_path: list, validity_path: list) -> tuple:
        """
        Compute the signatures of the two paths defining a discrepancy.

        Each path includes time as its first component (time augmentation),
        so the path lives in R^3 : (time, signal_1, signal_2).

        Args:
            values_path   : list of tuples (time, x1, x2)
            validity_path : list of tuples (time, v1, v2)
        Returns:
            (sig_values, sig_validity) : tuple of two np.arrays
            each of shape (13,) for d=3 and order=2
        """
        # Convert input lists to numpy arrays
        values_array   = np.array(values_path,   dtype=float)  # (T, 3)
        validity_array = np.array(validity_path, dtype=float)  # (T, 3)

        # Time augmentation already included as the first column
        # path = (time, signal_1, signal_2) lives in R^3
        sig_values   = self._compute_signature(values_array)
        sig_validity = self._compute_signature(validity_array)

        return sig_values, sig_validity

    def compute_point(self, values_path: list, validity_path: list) -> np.ndarray:
        """
        Compute the full point representation of a discrepancy for OT.

        Concatenates t_start, sig_values and sig_validity into a single vector
        that serves as the point in the OT coreset selection step.

        Args:
            values_path   : list of tuples (time, x1, x2)
            validity_path : list of tuples (time, v1, v2)
        Returns:
            np.array of shape (27,) for d=3 and order=2
        """
        sig_values, sig_validity = self.compute(values_path, validity_path)
        t_start = np.array([values_path[0][0]])
        return np.concatenate([t_start, sig_values, sig_validity])
    
    def process(
        self,
        discrepancies: List[Tuple]
    ) -> Tuple[List[Tuple[list, list]], List[int]]:
        points = []
        original_indices = []
        n = len(discrepancies)
        for i, (id, values_path, validity_path) in enumerate(discrepancies):
            printProgressBar(i+1, n, prefix='Progress', suffix='Processed signatures', length=50)

            point = self.compute_point(values_path, validity_path)
            points.append(point)
            original_indices.append(id)
        return np.array(points), np.array(original_indices)
    
    def export(
        self,
        points: np.ndarray,
        original_indices: List,
        output_path: str
    ) -> None:
        """
        Export computed signature points to a CSV file.

        Each row corresponds to one discrepancy, represented as its
        concatenated signature point (t_start + sig_values + sig_validity).

        Columns:
            discrepancy_id | dim_0 | dim_1 | ... | dim_26

        Args:
            points      : list of np.arrays of shape (27,),
                            output of compute_point() for each discrepancy
            output_path : path to the output CSV file
        """
        n_dims = points.shape[1]

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(['discrepancy_id'] + [f'dim_{i}' for i in range(n_dims)])

            for idx, point in zip(original_indices, points):
                writer.writerow([idx] + point.tolist())

        print(f"Exported {len(points)} signature points to {output_path}")
        
    def extract(self, input_path):
        df = pd.read_csv(input_path, sep=";")

        result = []
        for id_, group in df.groupby("discrepancy_id"):
            values = list(zip(group["t"], group["ve1"], group["ve2"]))
            validity = list(zip(group["t"], group["vy1"], group["vy2"]))
            result.append((id_, values, validity))
            
        return result