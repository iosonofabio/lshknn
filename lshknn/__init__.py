import numpy as np
import pandas as pd
from ._lshknn import knn_from_signature


class Lshknn:

    def __init__(
            self,
            data,
            similarity_k=20,
            similarity_threshold=0.2,
            m=100,
            ):

        self.data = data
        self.similarity_k = similarity_k
        self.similarity_threshold = similarity_threshold
        self.m = m
        self.n = data.shape[1]

    def _check_input(self):
        if self.m < 1:
            raise ValueError('m should be 1 or more')
        if self.graph_k < 1:
            raise ValueError('graph_k should be 1 or more')
        if self.similarity_k < 1:
            raise ValueError('similarity_k should be 1 or more')
        if (self.graph_threshold < 0) or (self.graph_threshold > 1):
            raise ValueError('graph_threshold should be between 0 and 1')
        if (self.similarity_threshold < 0) or (self.similarity_threshold > 1):
            raise ValueError('similarity_threshold should be between 0 and 1')
        if np.min(self.data.shape) < 2:
            raise ValueError('data should be at least 2x2 in shape')

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

        # TODO: Convert to memory blocks (64 bits)
        n_ints = (self.m - 1) // 64
        base = 2**np.arange(64).astype(np.uint64)
        ints = []
        for row in signature:
            for i in range(n_ints):
                sig = signature[i * 64: (i+1) * 64]
                ints.append(np.dot(sig, base[:len(sig)]))

        self.signature = np.concatenate(ints)

    def _knnlsh(self):
        if not hasattr(self, 'planes'):
            raise AttributeError('Compute signature first!')

        # NOTE: I allocate the output array in Python for ownership purposes
        knn = np.zeros((self.n, self.similarity_k), dtype=np.uint64)
        similarity = np.zeros((self.n, self.similarity_k), dtype=np.float64)
        n_neighbors = np.zeros(self.n, dtype=np.uint64)
        knn_from_signature(
                self.signature,
                knn,
                similarity,
                n_neighbors,
                self.m,
                self.similarity_k,
                self.similarity_threshold,
                )

        return knn, similarity, n_neighbors

    def __call__(self):
        self.check_input()
        self._normalize_data()
        self._generate_planes()
        self._compute_signature()
        return self._knnlsh()
