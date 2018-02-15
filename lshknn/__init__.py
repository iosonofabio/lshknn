import numpy as np
import pandas as pd
from ._lshknn import knn_from_signature


class Lshknn:

    def __init__(
            self,
            data,
            k=20,
            threshold=0.5,
            m=100,
            ):

        self.data = data
        self.k = k
        self.threshold = threshold
        self.m = m

    def _normalize_data(self):
        # Substract average across genes for each cell
        # FIXME: preserve sparsity?!
        self.data -= self.data.mean(axis=0)

    def _generate_planes(self):
        self.planes = np.random.normal(
                loc=0,
                scale=1,
                size=(self.data.shape[0], self.m),
                )

    def _compute_signature(self):
        if not hasattr(self, 'planes'):
            raise AttributeError('Generate planes first!')

        signature = np.dot(self.data.T, self.planes) > 0

        # TODO: Convert to memory blocks
        self.signature = signature


    def _knnlsh(self):
        if not hasattr(self, 'planes'):
            raise AttributeError('Compute signature first!')

        # TODO: plug pybind11 CPython wrap
        knn = np.random.randint(
                low=0,
                high=self.data.shape[1],
                size=(self.data.shape[1], self.k),
                )

        return knn

    def __call__(self):
        self._normalize_data()
        self._generate_planes()
        self._compute_signature()
        return self._knnlsh()
