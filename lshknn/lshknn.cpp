//#include <iostream>
#include <map>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

using MatrixXdR = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Struct for Sorting cells by similarity
class SimilarCell {
public:
    double similarity;
    uint64_t cell;

    // Constructors.
    SimilarCell() : similarity(0), cell(0) {}
    SimilarCell(double similarity, uint64_t cell) : similarity(similarity), cell(cell) {}

    // Comparisons for unique
    bool operator==(const SimilarCell& other){ return this->cell == other.cell; }
    bool operator!=(const SimilarCell& other){ return !(*this == other); }

};

// Compare function to  sort SimilarCells
bool compareSimilarCells(SimilarCell i, SimilarCell j) { return (i.similarity > j.similarity); }

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

// Fill knn output matrix
void fillOutputMatrices(
    py::EigenDRef<Eigen::Matrix<uint64_t, -1, -1> > knn,
    py::EigenDRef<Eigen::Matrix<double, -1, -1> > similarity,
    py::EigenDRef<Eigen::Matrix<uint64_t, -1, -1> > nNeighbors,
    uint64_t k,
    std::vector<SimilarCell>& candidates,
    uint64_t nNeighborsCell,
    uint64_t cellFocal) {

    // Fill output matrix
    uint64_t knni = 0;
    for(std::vector<SimilarCell>::iterator knnit=candidates.begin();
        (knnit != candidates.end()) && (knni < (uint64_t)k);
        knnit++) {
        knn(cellFocal, knni) = knnit->cell;
        similarity(cellFocal, knni) = knnit->similarity;
        knni++;
    }
    nNeighbors(0, cellFocal) = nNeighborsCell;

}

// Compute k nearest neighbors and similarity values naively, i.e.
// iterating over all pairs
void computeNeighborsViaAllPairs(
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, -1> > signature,
    py::EigenDRef<Eigen::Matrix<uint64_t, -1, -1> > knn,
    py::EigenDRef<Eigen::Matrix<double, -1, -1> > similarity,
    py::EigenDRef<Eigen::Matrix<uint64_t, -1, -1> > nNeighbors,
    const int n,
    const int k,
    const size_t wordCount,
    std::vector<double>& similarityTable,
    const double threshold) {

    std::vector<SimilarCell> candidates;
    
    // Calculate all similarities with this cell
    for (uint64_t cell1=0; cell1 < (uint64_t)n; cell1++) {
        candidates.clear();
        BitSetPointer bp1(signature.data() + cell1 * wordCount, wordCount);
        for (uint64_t cell2=0; cell2 < (uint64_t)n; cell2++) {
            if (cell2 == cell1)
                continue;
            BitSetPointer bp2(signature.data() + cell2 * wordCount, wordCount);
            double sim = computeCellSimilarity(bp1, bp2, similarityTable);
            if (sim >= threshold) {
                candidates.push_back({ sim, cell2 });
                //std::cout << "cell1: " << cell1 << ", cell2: " << cell2 << ", sim: " << sim << "\n";
            }
        }

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

        // Fill holes
        uint64_t nNeighborsCell = candidates.size() < (uint64_t)k ? candidates.size() : k;
        candidates.resize(k, {-1, (uint64_t)n});

        // Fill output matrix
        fillOutputMatrices(knn, similarity, nNeighbors, k, candidates, nNeighborsCell, cell1);
    }

}

// Slice multibit signatures for LSH computation
// It is much easier if slices are 64 bit long or less; this is a fair
// requirement for our purposes as having more than 2**64 buckets is useless
uint64_t sliceSignature(uint64_t *data, uint64_t firstBit, uint64_t nBits) {
    if (nBits > 64) {
        throw std::runtime_error("Slices must be at most 64 bit long");
    } else if (nBits == 0) {
        throw std::runtime_error("Slices must be at least 1 bit long");
    }

    // Find whether we are crossing word boundary
    uint64_t firstWord = firstBit >> 6;
    uint64_t lastWord = (firstBit + nBits) >> 6;

    // Not crossing word boundary is easy, just shift and bitwise & and shift back
    if (lastWord == firstWord) {
        // e.g. if nBits = 3 and firstBit = 66, we go to the second word,
        // then shift = 2, so other is b11100 = 28
        // after the bitwise &, the first shift bits are all 0 anyway, we can
        // trash them to be consistent
        uint64_t shift = firstBit % 64;
        uint64_t other = ((1ULL << nBits) - 1ULL) << shift;
        return ((*(data + firstWord)) & other) >> shift;

    // Else, we have to split the job in two
    } else {
        uint64_t shift = firstBit % 64;
        uint64_t nBitsFirst = 64 - shift;
        uint64_t other = ((1ULL << nBitsFirst) - 1ULL) << shift;
        uint64_t out = ((*(data + firstWord)) & other) >> shift;

        // The second word has no shift
        uint64_t nBitsSecond = nBits - nBitsFirst;
        other = ((1ULL << nBitsSecond) - 1ULL) << 0;
        out |= ((*(data + lastWord)) & other) << nBitsFirst;
        return out;
    }

}

// Fill hash map of slice
void fillHashMap(
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, -1> > signature,
    const int n,
    const size_t wordCount,
    // Slices are at most 64 bits
    std::map<uint64_t, std::vector<uint64_t> >& sliceMap,
    const uint64_t firstBit,
    const uint64_t sliceLength) {

    // Clear hash map
    sliceMap.clear();

    // Iterate over cells
    for (uint64_t cell=0; cell < (uint64_t)n; cell++) {
        uint64_t *cellData = (uint64_t*)(signature.data()) + cell * wordCount;
        uint64_t cellHash = sliceSignature(cellData, firstBit, sliceLength);
        sliceMap[cellHash].push_back(cell);
    }
}

// Compute k nearest neighbors via signature slices
void computeNeighborsViaSlices(
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, -1> > signature,
    py::EigenDRef<Eigen::Matrix<uint64_t, -1, -1> > knn,
    py::EigenDRef<Eigen::Matrix<double, -1, -1> > similarity,
    py::EigenDRef<Eigen::Matrix<uint64_t, -1, -1> > nNeighbors,
    const int n,
    const int k,
    const size_t wordCount,
    std::vector<double>& similarityTable,
    const double threshold,
    const uint64_t sliceLength,
    const size_t m) {

    // Candidate neighbors
    std::vector< std::vector<SimilarCell> > neighbors(n);

    // Hash map
    std::map<uint64_t, std::vector<uint64_t> > sliceMap;

    // Iterate over slices
    uint64_t nSlices = m / sliceLength;
    for (uint64_t slice=0; slice < nSlices; slice++) {

        // Make hash map for all cells
        fillHashMap(signature, n, wordCount, sliceMap, slice * sliceLength, sliceLength);

        // Iterate over map and calculate similarities within each bucket
        for(std::map<uint64_t, std::vector<uint64_t> >::iterator mit=sliceMap.begin();
            mit != sliceMap.end(); mit++) {
            
            // Naive algorithm, recalculate distances from each cell
            std::vector<uint64_t> cellsBucket = mit->second;
            std::vector<SimilarCell> candidates;
            for(std::vector<uint64_t>::iterator cit=cellsBucket.begin();
                cit != cellsBucket.end();
                cit++) {

                candidates.clear();
                BitSetPointer bp1(signature.data() + (*cit) * wordCount, wordCount);
                
                for(std::vector<uint64_t>::iterator cit2=cellsBucket.begin();
                    cit2 != cellsBucket.end();
                    cit2++) {
                    if (cit2 != cit) {
                        BitSetPointer bp2(signature.data() + (*cit2) * wordCount, wordCount);
                        double sim = computeCellSimilarity(bp1, bp2, similarityTable);
                        if (sim >= threshold)
                            candidates.push_back({ sim, *cit2 });
                    }
                }

                // Sort candidates and take top k neighbors
                std::sort(candidates.begin(), candidates.end(), compareSimilarCells);
                if (candidates.size() > (uint64_t)k)
                    candidates.resize(k);
                neighbors[*cit].insert(neighbors[*cit].end(), candidates.begin(), candidates.end());
            }
        }
    }

    // Take top k candidates for each cell
    uint64_t cellFocal = 0;
    for(std::vector< std::vector<SimilarCell> >::iterator nit=neighbors.begin();
        nit != neighbors.end();
        nit++) {

        // FIXME: optimize to only sort k of them, the k largest
        std::vector<SimilarCell> neighborsCell = *nit;
        std::sort(neighborsCell.begin(), neighborsCell.end(), compareSimilarCells);
        // Remove duplicates
        std::vector<SimilarCell>::iterator itLast = std::unique(neighborsCell.begin(), neighborsCell.end());
        neighborsCell.resize(std::distance(neighborsCell.begin(), itLast));

        // Fill holes
        uint64_t nNeighborsCell = neighborsCell.size() < (uint64_t)k ? neighborsCell.size() : k;
        neighborsCell.resize(k, {-1, (uint64_t)n});

        // Fill output matrix
        fillOutputMatrices(knn, similarity, nNeighbors, k, neighborsCell, nNeighborsCell, cellFocal);

        cellFocal++;
    }
}

///////////////////////////////////////////////////////////
// Python Interface
///////////////////////////////////////////////////////////
void knn_from_signature(
    py::EigenDRef<const Eigen::Matrix<uint64_t, -1, -1> > signature,
    py::EigenDRef<Eigen::Matrix<uint64_t, -1, -1> > knn,
    py::EigenDRef<Eigen::Matrix<double, -1, -1> > similarity,
    py::EigenDRef<Eigen::Matrix<uint64_t, -1, -1> > nNeighbors,
    const int n,
    const int m,
    const int k,
    const double threshold,
    const int sliceLength) {

    // signature is a vector containing 64 bit integers for all n cells, the number of 64bit for each cell is
    size_t wordCount = 1 + ((m - 1) / 64);
    // so we can parse 0 to (wordCount - 1) for cell 1, wordCount to (2 * wordCount - 1) for cell 2, etc.

    // Compute the similarity table with m bits
    std::vector<double> similarityTable;
    computeSimilarityTable((size_t)m, similarityTable);

    // Slower version, go through n^2 pairs
    if (sliceLength == 0) {
        computeNeighborsViaAllPairs(
        signature, knn, similarity, nNeighbors,
        n, k, wordCount,
        similarityTable,
        threshold);

    // Faster version
    } else {
        // 1. Make non-overlapping q bit slices, total m / q
        //    Up to 2^q hashes per group
        // 2. For each subgroup, hash cells
        // 3.     For each cell, find k neighbors in the same hash group
        // 4. Sort cell neighbours from all subgroups and take first k
        // 5. Format for returning
        computeNeighborsViaSlices(
            signature, knn, similarity, nNeighbors,
            n, k, wordCount,
            similarityTable,
            threshold,
            sliceLength,
            (size_t)m);
    }
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
