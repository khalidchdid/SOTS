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
from tqdm import tqdm
import pandas as pd


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
    
    def process(
        self,
        digits: np.ndarray
    ) -> np.ndarray:
        signatures = []
        original_indices = []
        for i, digit in tqdm(enumerate(digits)):

            signature = self._compute_signature(digit.T)
            signatures.append(signature)
            original_indices.append(i)
        return np.array(signatures), np.array(original_indices)