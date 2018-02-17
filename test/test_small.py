import numpy as np
import lshknn

data = np.array(
        [[1, 0, 1, 0],
         [0, 1, 0, 1],
         ], dtype=np.float64,
        )

c = lshknn.Lshknn(
        data=data,
        k=1,
        threshold=0.2,
        m=10,
        )

knn, similarity, n_neighbors = c()
assert (knn == [[2], [3], [0], [1]]).all()
