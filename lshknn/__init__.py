import numpy as np
from ._lshknn import knn_from_signature


class Lshknn:
    '''Local sensitive hashing k-nearest neighbors

    Calculate k-nearest neighbours of vectors in a high dimensional space
    using binary signatures of random hyperplanes as a coarse grain vector
    representation.

    Usage:

    >>> # Matrix with two features and three samples
    >>> data = np.array([[0, 1, 2], [3, 2, 1]])
    >>> knn, similarity, n_neighbors = Lshknn(data=data, k=1)()
    '''

    def __init__(
            self,
            data,
            k=20,
            threshold=0.2,
            m=100,
            slice_length=None,
            ):
        '''Local sensitive hashing k-nearest neighbors.

        Arguments:
            data (dense or sparse matrix or dataframe): the \
vectors to analyze. Shape is (f, n) with f features and n samples.
            k (int): number of neighbors to find.
            threshold (float): minimal correlation threshold to be considered \
a neighbor.
            m (int): number of random hyperplanes to use.
            slice_length (int or None): The slice length to use if \
approximating signatures in hash buckets.

        Returns:
            (knn, similarity, n_neighbors): triple with the nearest \
neighbors, the matrix of similarities, and the number of neighbors for each \
cell.
        '''

        self.data = data
        self.k = k
        self.threshold = threshold
        self.m = m
        self.n = data.shape[1]
        self.slice_length = slice_length

    def _check_input(self):
        if not isinstance(self.data, np.ndarray):
            raise ValueError('data should be a numpy 2D matrix')
        if len(self.data.shape) != 2:
            raise ValueError('data should be a 2D matrix')
        if np.min(self.data.shape) < 2:
            raise ValueError('data should be at least 2x2 in shape')
        if self.m < 1:
            raise ValueError('m should be 1 or more')
        if self.k < 1:
            raise ValueError('k should be 1 or more')
        if (self.threshold < -1) or (self.threshold > 1):
            raise ValueError('threshold should be between -1 and 1')
        if (self.slice_length is not None) and (self.slice_length < 1):
            raise ValueError('slice_length should be None or between 1 and 64')
        if (self.slice_length is not None) and (self.slice_length > self.m):
            raise ValueError('slice_length cannot be longer than m')

    def _normalize_input(self):
        if self.slice_length is None:
            self.slice_length = 0

        # Substract average across genes for each cell
        # FIXME: preserve sparsity?!
        self.data = self.data - self.data.mean(axis=0)

    def _generate_planes(self):
        # Optimization flags
        if self.data.flags['C_CONTIGUOUS']:
            planes = np.random.normal(
                    loc=0,
                    scale=1,
                    size=(self.m, self.data.shape[0]),
                    ).T
            planes /= planes.sum(axis=0)
            planes = planes.T
        else:
            planes = np.random.normal(
                    loc=0,
                    scale=1,
                    size=(self.data.shape[0], self.m),
                    )
            planes /= planes.sum(axis=0)
        self.planes = planes

    def _compute_signature(self):
        if not hasattr(self, 'planes'):
            raise AttributeError('Generate planes first!')

        # NOTE: 90% of the time is spent here
        if self.data.flags['C_CONTIGUOUS']:
            signature = np.dot(self.planes, self.data).T > 0
            signature = np.ascontiguousarray(signature)
        else:
            signature = np.dot(self.data.T, self.planes) > 0

        word_count = 1 + (self.m - 1) // 64
        base = 1 << np.arange(64, dtype=np.uint64)
        ints = []
        for row in signature:
            for i in range(word_count):
                sig = row[i * 64: (i+1) * 64]
                ints.append(np.dot(sig, base[:len(sig)]))

        self.signature = np.array([ints], np.uint64)

    def _knnlsh(self):
        if not hasattr(self, 'planes'):
            raise AttributeError('Compute signature first!')

        # NOTE: I allocate the output array in Python for ownership purposes
        self.knn = self.n * np.ones((self.n, self.k), dtype=np.uint64)
        self.similarity = np.zeros((self.n, self.k), dtype=np.float64)
        self.n_neighbors = np.zeros((self.n, 1), dtype=np.uint64)
        knn_from_signature(
                self.signature,
                self.knn,
                self.similarity,
                self.n_neighbors,
                self.n,
                self.m,
                self.k,
                self.threshold,
                self.slice_length,
                )

    def _format_output(self):
        # Kill empty spots in the matrix
        # Note: this may take a while compared to the algorithm
        ind = self.knn >= self.n
        self.knn = np.ma.array(self.knn, mask=ind, copy=False)
        self.similarity = np.ma.array(self.similarity, mask=ind, copy=False)
        return self.knn, self.similarity, self.n_neighbors

    def __call__(self):
        self._check_input()
        self._normalize_input()
        self._generate_planes()
        self._compute_signature()
        self._knnlsh()
        return self._format_output()
