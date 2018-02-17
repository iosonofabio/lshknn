//#include <iostream>
#include <map>
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
    const uint64_t* begin;
    const uint64_t* end;

    // Constructors.
    BitSetPointer(const uint64_t* begin=0, const uint64_t* end=0) : begin(begin), end(end) {}
    BitSetPointer(const uint64_t* begin, uint64_t wordCount) : begin(begin), end(begin+wordCount) {}

    // Return the number of 64-bit words in the bit set.
    uint64_t wordCount() const { return end - begin; }

};

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

struct SimilarCell {
    double similarity;
    uint64_t cell;
};

bool compareSimilarCells(SimilarCell i, SimilarCell j) { return (i.similarity > j.similarity); }

void computeNeighborsViaAllPairs(
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, -1> > signature,
    py::EigenDRef<Eigen::Matrix<uint64_t, -1, -1> > knn,
    py::EigenDRef<Eigen::Matrix<double, -1, -1> > similarity,
    const int n,
    const int k,
    const size_t wordCount,
    std::vector<double>& similarityTable,
    const double threshold) {

    std::vector<SimilarCell> candidates(n);
    for (uint64_t cell1=0; cell1 < (uint64_t)n; cell1++) {

	// Calculate all similarities with this cell
        for (uint64_t cell2=0; cell2 < (uint64_t)n; cell2++) {
		BitSetPointer bp1(signature.data() + cell1 * wordCount, wordCount);
		BitSetPointer bp2(signature.data() + cell2 * wordCount, wordCount);
		double sim = computeCellSimilarity(bp1, bp2, similarityTable);
		candidates[cell2] = { sim, cell2 };

		//std::cout << "cell1: " << cell1 << ", cell2: " << cell2 << ", sim: " << sim << "\n";
        }
	candidates[cell1] = { -1, cell1};

	// Sort cells by similarities
	std::sort(candidates.begin(), candidates.end(), compareSimilarCells);

	/*
	std::cout << "cell1 neighbors sorted for cell " << cell1 << ":\n";
        for (std::vector<SimilarCell>::iterator it=candidates.begin();
	     it != candidates.end(); it++) {
		std::cout << (it->cell);
	}
	std::cout << "\n";
        */

	// Fill output matrix
	for(uint64_t neigh=0; neigh < (uint64_t)k; neigh++) {
	    if(candidates[neigh].similarity >= threshold) {
	        knn(cell1, neigh) = candidates[neigh].cell;
	        similarity(cell1, neigh) = candidates[neigh].similarity;
	    } else {
		// n is above the max, so it is used as nans
	        knn(cell1, neigh) = n;
	        similarity(cell1, neigh) = 0;
	    }
	}
    }

}

/*
Eigen::MatrixXd computeHashSlices(
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, -1> > signature,
    const int n,
    const int q,
    const int g,
    const size_t wordCount,
    Eigen::Matrix<uint64_t, Dynamic, 1>& knn_tmp,
    Eigen::Matrix<double, Dynamic, 1>& sim_tmp,
    std::vector<double>& similarityTable,
    ) {

    // Hash map
    std::map<BitSetPointer, std::vector<uint64_t> > signatureMap;

    // Iterate over slices
    for (int gi=0; gi < g; gi++) {
        // Clear signatureMap
        signatureMap.clear();

        // Iterate over cells
	for (uint64_t cell=0; cell < n; cell++) {
            BitSetPointer bp(
                begin=signature.data() + cell * wordCount + gi * q,
		wordCount=q);
	    signatureMap[bp].push_back(cell);
        }

	// Now we have the first sliced signatures
	// Let's add neighbor candidates
        for (std::map<BitSetPointer, std::vector<uint64_t> >::iterator it=signatureMap.begin();
             it!=mymap.end();
             it++) {
             // Calculate all pairwise distances within group
	     int ng = it->second.size();
	     MatrixXdR similarityHashGroup(ng, ng);
	     for (std::vector<int>::iterator vit = it->second.begin(), int cell1=0; vit != it->second.end(); vit++, cell1++) {
                 for (std::vector<int>::iterator vit2 = it->second.begin(), int cell2=0; vit2 != it->second.end(); vit2++, cell2++) {
                     similarityHashGroup(cell1, cell2) = computeCellSimilarity(
				     BitSetPointer(begin=signature.data() + (*vit1) * wordCount, wordCount=wordCount),
				     BitSetPointer(begin=signature.data() + (*vit2) * wordCount, wordCount=wordCount),
				     )
	          }
	     }
             
	     // Sort rows and take top 20
    }


    for(CellId cellId=0; cellId<cellCount; cellId++) {
        signatureMap[lsh.getSignature(cellId)].push_back(cellId);
}
    for(int i=0; i < n; i++) {
    
    }

}
*/

///////////////////////////////////////////////////////////
// Python Interface
///////////////////////////////////////////////////////////
void knn_from_signature(
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, -1> > signature,
    py::EigenDRef<Eigen::Matrix<uint64_t, -1, -1> > knn,
    py::EigenDRef<Eigen::Matrix<double, -1, -1> > similarity,
    py::EigenDRef<Eigen::Matrix<uint64_t, -1, -1> > n_neighbors,
    const int n,
    const int m,
    const int k,
    const double threshold) {

    // signature is a vector containing 64 bit integers for all n cells, the number of 64bit for each cell is
    size_t wordCount = 1 + ((m - 1) / 64);
    // so we can parse 0 to (wordCount - 1) for cell 1, wordCount to (2 * wordCount - 1) for cell 2, etc.

    // Compute the similarity table with m bits
    std::vector<double> similarityTable;
    computeSimilarityTable((size_t)m, similarityTable);

    // Slower version, go through n^2 pairs
    computeNeighborsViaAllPairs(
	signature, knn, similarity,
	n, k, wordCount,
	similarityTable,
	threshold);
    
    // Faster version
    //// 1. Create g bit subgroups with q 64-bit words each
    ////    Up to 2^q hashes per group
    //int q = 1;
    //int g = 1 + (m - 1) / q;
    //Eigen::Matrix<uint64_t, Dynamic, 1> knn_tmp(n, q * g);
    //Eigen::Matrix<double, Dynamic, 1> sim_tmp(n, q * g);
    //Eigen::MatrixXd knn_tmp = computeHashSlices(
    //    signature,
    //    n, q, g, wordCount,
    //    knn_tmp,
    //    sim_tmp,
    //    similarityTable);
    //// 2. For each subgroup, hash cells
    //// 3.     For each cell, find k neighbors in the same hash group
    //// 4. Sort cell neighbours from all subgroups and take first k
    //// 5. Format for returning
	
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
