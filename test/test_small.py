import numpy as np
import lshknn

data = np.array(
        [[1, 0, 1, 0],
         [0, 1, 0, 1],
         ], dtype=np.float64,
        )

c = lshknn.Lshknn(
        data=data,
        graph_k=1,
        similarity_k=1,
        graph_threshold=0.5,
        simlarity_threshold=0.2,
        m=10,
        )

knn, similarity = c()
assert (knn == [[2], [3], [0], [2]])
