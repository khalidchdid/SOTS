"""
Input: a list of discrepancies where each discrepancy is a tuple of lists
Output: a list of discrepancies where each discrepancy is two lists of tuples (t, sve, ove) and (t, svy, ovy)

Preprocesses raw discrepancies into aligned path tuples ready for
    signature computation.

    Each raw discrepancy is a tuple (t1, ve1, vy1, t2, ve2, vy2) where:
        t1, t2   : lists of timestamps (potentially different sizes)
        ve1, ve2 : lists of values at each timestamp
        vy1, vy2 : lists of validity at each timestamp

    The preprocessing pipeline:
        1. Build a common timeline t = union(t1, t2), sorted
        2. Interpolate each signal onto t using a step function
           (value at t_k held constant until t_{k+1})
        3. Build two lists of tuples:
               values_path   = [(t, ve1, ve2), ...]
               validity_path = [(t, vy1, vy2), ...]

Potentially applying different imputation strategies other than linear interpolation

cf. docs/papers/ Path Imputation Strategies ...

"""

import numpy as np
from typing import List, Tuple
import csv

from modules.utilities.terminal_style import *


class Interpolator:

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    @staticmethod
    def _common_timeline(t1: list, t2: list) -> np.ndarray:
        """
        Build the common timeline as the sorted union of t1 and t2.

        Args:
            t1 : list of timestamps for signal 1
            t2 : list of timestamps for signal 2
        Returns:
            np.array of sorted unique timestamps
        """
        return np.union1d(t1, t2)

    @staticmethod
    def _step_interpolate(t_orig: list, values: list, t_common: np.ndarray) -> np.ndarray:
        """
        Interpolate a signal onto a common timeline using a step function.

        For each timestamp in t_common, the value is the last observed
        value at or before that timestamp (i.e. x_{t_k} held constant
        until t_{k+1}).

        Args:
            t_orig   : original list of timestamps
            values   : list of values at each original timestamp
            t_common : common timeline to interpolate onto
        Returns:
            np.array of interpolated values, shape (len(t_common),)
        """
        t_orig  = np.array(t_orig)
        values  = np.array(values)
        result  = np.zeros(len(t_common))

        for i, t in enumerate(t_common):
            # Find the index of the last original timestamp <= t
            idx = np.searchsorted(t_orig, t, side='right') - 1
            idx = max(idx, 0) # if t is before the timestamp, use the first value
            result[i] = values[idx]

        return result

    def _align_discrepancy(
        self,
        t1: list, ve1: list, vy1: list,
        t2: list, ve2: list, vy2: list,
    ) -> Tuple[list, list]:
        """
        Align a single discrepancy onto a common timeline.

        Args:
            t1, ve1, vy1 : timestamps, values, validity for signal 1
            t2, ve2, vy2 : timestamps, values, validity for signal 2
        Returns:
            values_path   : list of tuples (t, ve1, ve2)
            validity_path : list of tuples (t, vy1, vy2)
        """
        if len(t1) == 0 or len(t2) == 0:
            return None

        # Step 1 — Build common timeline
        t_common = self._common_timeline(t1, t2)

        # Step 2 — Interpolate all signals onto common timeline
        ve1_interp = self._step_interpolate(t1, ve1, t_common)
        ve2_interp = self._step_interpolate(t2, ve2, t_common)
        vy1_interp = self._step_interpolate(t1, vy1, t_common)
        vy2_interp = self._step_interpolate(t2, vy2, t_common)

        # Step 3 — Build output lists of tuples
        values_path   = [(t.item(), ve1_interp[i].item(), ve2_interp[i].item()) for i, t in enumerate(t_common)]
        validity_path = [(t.item(), vy1_interp[i].item(), vy2_interp[i].item()) for i, t in enumerate(t_common)]

        return values_path, validity_path

    # ------------------------------------------------------------------
    # Public method
    # ------------------------------------------------------------------

    def process(
        self,
        discrepancies: List[Tuple]
    ) -> List[Tuple[list, list]]:
        """
        Process a list of raw discrepancies into aligned path tuples.

        Args:
            discrepancies : list of tuples (t1, ve1, vy1, t2, ve2, vy2)
        Returns:
            list of tuples (values_path, validity_path) where:
                values_path   : list of tuples (t, ve1, ve2)
                validity_path : list of tuples (t, vy1, vy2)
        """
        aligned = []
        n = len(discrepancies)
        for i, (id, t1, ve1, vy1, t2, ve2, vy2) in enumerate(discrepancies):
            printProgressBar(i+1, n, prefix='Progress', suffix='Interpolated_discrepancies', length=50)
            
            result = self._align_discrepancy(
                t1, ve1, vy1,
                t2, ve2, vy2
            )
            if result is not None:
                values_path, validity_path = result
                aligned.append((id, values_path, validity_path))
        return aligned
    
    def export(
        self,
        aligned: List[Tuple[list, list]],
        output_path: str = "aligned_discrepancies.csv"
    ) -> None:
        """
        Export aligned discrepancies to a CSV file.

        Each row corresponds to one timestep of one discrepancy.
        Discrepancies are separated by a blank row.

        Columns:
            discrepancy_id | t | ve1 | ve2 | vy1 | vy2

        Args:
            aligned     : output of process()
            output_path : path to the output CSV file
        """
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(['discrepancy_id', 't', 've1', 've2', 'vy1', 'vy2'])

            for id, values_path, validity_path in aligned:
                for (t, ve1, ve2), (_, vy1, vy2) in zip(values_path, validity_path):
                    writer.writerow([id, t, ve1, ve2, vy1, vy2])

        print(f"Exported {len(aligned)} discrepancies to {output_path}")
