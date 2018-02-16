#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

using MatrixXdR = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Class that stores pointers to the begin and end of a bit set.
// Does not own the memory, and copies are shallow copies.
// Implements low level operations on bit sets.
class BitSetPointer {
public:

    // Begin and and pointers of the bit set.
    uint64_t* begin;
    uint64_t* end;

    // Constructors.
    BitSetPointer(uint64_t* begin=0, uint64_t* end=0) : begin(begin), end(end) {}
    BitSetPointer(uint64_t* begin, uint64_t wordCount) : begin(begin), end(begin+wordCount) {}

    // Return the number of 64-bit words in the bit set.
    uint64_t wordCount() const { return end - begin; }

}


// Count the number of mismatching bits between two bit vectors.
// Use SSE 4.2 builtin instruction popcount
inline uint64_t countMismatches(
    const BitSetPointer& x,
    const BitSetPointer& y)
{
    const uint64_t wordCount = x.wordCount();
    uint64_t mismatchCount = 0;
    for(uint64_t i = 0; i < wordCount; i++) {
        mismatchCount += __builtin_popcountll(x.begin[i] ^ y.begin[i]);
    }
    return mismatchCount;
}

// Compute the LSH similarity between two cells,
// specified by their ids local to the cell set used by this Lsh object.
double computeCellSimilarity(
    const BitSetPointer signature0,
    const BitSetPointer signature1,
    std::vector<double>& similarityTable)
{
    // Count the number of bits where the signatures of these two cells disagree.
    const size_t mismatchingBitCount = countMismatches(signature0, signature1);

    // Return the similarity corresponding to this number of mismatching bits.
    return similarityTable[mismatchingBitCount];
}

// Compute the similarity (cosine of the angle) corresponding to each number of mismatching bits.
void computeSimilarityTable(
    const size_t lshCount,
    std::vector<double>& similarityTable)
{
    // Initialize the similarity table.
    similarityTable.resize(lshCount + 1);

    // Loop over all possible numbers of mismatching bits.
    for(size_t mismatchingBitCount = 0;
        mismatchingBitCount <= lshCount;
	mismatchingBitCount++) {

        // Compute the angle between the vectors corresponding to
        // this number of mismatching bits.
        const double angle = double(mismatchingBitCount) *
            3.14159265359 / double(lshCount);

        // The cosine of the angle is the similarity for
        // this number of mismatcning bits.
        similarityTable[mismatchingBitCount] = std::cos(angle);
    }

}

///////////////////////////////////////////////////////////
// Python Interface
///////////////////////////////////////////////////////////
void knn_from_signature(
    py::EigenDRef<Eigen::MatrixXd> signature,
    py::EigenDRef<Eigen::MatrixXd> knn,
    py::EigenDRef<Eigen::MatrixXd> similarity,
    py::EigenDRef<Eigen::MatrixXd> n_neighbors,
    const int n,
    const int m,
    const int k,
    const double threshold) {

    // signature is a vector containing 64 bit integers for all n cells, the number of 64bit for each cell is
    // so we can parse 0 to (b - 1) for cell 1, b to (2b - 1) for cell 2, etc.
    size_t wordCount = 1 + ((m - 1) / 64);

    // Compute the similarity table with m bits
    std::vector<double> similarityTable;
    computeSimilarityTable((size_t)m, similarityTable);
    
    //TODO: write the actual code!
    // We gotta wrap chunks of signature as non-memory-owning BitSetPointer to keep some resemblance to Paolo's
    // original API
    //TODO
    for(int i=0; i < n; i++) {
        if (i == 0)
    	    knn(i, 0) = 4;
        else
    	    knn(i, 0) = 0;
    }
    // end TODO
	
}

PYBIND11_MODULE(_lshknn, m) {
    m.def("knn_from_signature", &knn_from_signature, R"pbdoc(
        Add to an existing matrix.

        This tests whether we can access numpy array from Eigen.
        Some other explanation about the add function.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
