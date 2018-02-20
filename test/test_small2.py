import numpy as np
import lshknn

threshold = 0.2
k = 4

data = np.array(
        # 0  1  2   3     4     5     6   7
        [[1, 0, 1, 0.19, 0.25, 0.35, 0.7, 0],
         [0, 1, 0, 0.81, 0.75, 0.65, 0.3, 1],
         [0, 0, 0, 0.00, 0.00, 0.00, 0.0, 0],
         [0, 0, 0, 0.00, 0.00, 0.00, 0.0, 0],
         [0, 0, 0, 0.00, 0.00, 0.00, 0.0, 0],
         [0, 0, 0, 0.00, 0.00, 0.00, 0.0, 0],
         [0, 0, 0, 0.00, 0.00, 0.00, 0.0, 0],
         [0, 0, 0, 0.00, 0.00, 0.00, 0.0, 0],
         ], dtype=np.float64,
        )

corr = []
for c1 in data.T:
    corr.append([])
    for c2 in data.T:
        r = np.dot(c1 - c1.mean(), c2 - c2.mean())
        r /= np.sqrt(((c1 - c1.mean())**2).sum() * ((c2 - c2.mean())**2).sum())
        corr[-1].append(r)
corr = np.ma.masked_less(corr, threshold)
corr[np.arange(corr.shape[0]), np.arange(corr.shape[0])] = np.ma.masked
knn_sol = []
for row in corr:
    knn_sol.append(np.ma.argsort(1 - row)[:k])
knn_sol = np.ma.array(knn_sol)
similarity_sol = []
for i, row in enumerate(knn_sol):
    similarity_sol.append(corr[i][row])
similarity_sol = np.ma.array(similarity_sol)
knn_sol.mask = similarity_sol.mask

print('Test not so small data without slicing')
c = lshknn.Lshknn(
        data=data,
        k=k,
        threshold=threshold,
        m=80,
        slice_length=None,
        )
knn, similarity, n_neighbors = c()
assert np.abs((similarity - similarity_sol).mean()) < 0.1
print('Done')

print('Test not so small data with slicing')
c = lshknn.Lshknn(
        data=data,
        k=k,
        threshold=threshold,
        m=80,
        slice_length=6,
        )
knn, similarity, n_neighbors = c()
assert np.abs((similarity - similarity_sol).mean()) < 0.1
print('Done')
