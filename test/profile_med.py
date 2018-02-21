import time
import numpy as np
import lshknn

threshold = 0.2
k = 5

n_genes = 10000
n_cells = 2000
data = np.random.rand(n_genes, n_cells)


t0 = time.time()
c = lshknn.Lshknn(
        data=data,
        k=k,
        threshold=threshold,
        m=128,
        slice_length=None,
        )
knn, similarity, n_neighbors = c()
t1 = time.time()
print('No slicing time for a dataset of {:} genes and {:} cells: {:.2f} secs.'.format(
    n_genes,
    n_cells,
    t1 - t0),
    )


t0 = time.time()
c = lshknn.Lshknn(
        data=data,
        k=k,
        threshold=threshold,
        m=128,
        slice_length=8,
        )
knn, similarity, n_neighbors = c()
t1 = time.time()
print('Slicing time for a dataset of {:} genes and {:} cells: {:.2f} secs.'.format(
    n_genes,
    n_cells,
    t1 - t0),
    )
