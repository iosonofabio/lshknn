[![Build Status](https://travis-ci.org/iosonofabio/lshknn.svg?branch=master)](https://travis-ci.org/iosonofabio/lshknn)

# LSHKNN
CPython module for fast calculation of k nearest neighbor (KNN) graphs in high-dimensional vector spaces using Pearson correlation distance and local sensitive hashing (LSH).

The current application is analysis of single cell RNA-Seq data. Paolo Carnevali @ Chan Zuckerberg Initiative is the owner of the algorithm code, which is also under MIT license:

https://github.com/chanzuckerberg/ExpressionMatrix2

## Usage
```python
import numpy as np
import lshknn

# Make mock data
# 2 features (rows), 4 samples (columns)
data = np.array(
        [[1, 0, 1, 0],
         [0, 1, 0, 1]],
        dtype=np.float64)

# Instantiate class
c = lshknn.Lshknn(
        data=data,
        k=1,
        threshold=0.2,
        m=10,
        slice_length=4)

# Call subroutine
knn, similarity, n_neighbors = c()

# Check result
assert (knn == [[2], [3], [0], [1]]).all()
```
