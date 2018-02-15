#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

using MatrixXdR = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

int knn_from_signature(
		py::EigenDRef<Eigen::MatrixXd> signature,
		py::EigenDRef<Eigen::MatrixXd> knn,
		py::EigenDRef<Eigen::MatrixXd> similarity,
		py::EigenDRef<Eigen::MatrixXd> n_neighbors,
		int n, int m, int k, threshold) {

	// signature is a vector containing 64 bit integers for all n cells, the number of integers for each cell is
	// b = (1 + ((m - 1) div 64))
	// so we can parse 0 to (b - 1) for cell 1, b to (2b - 1) for cell 2, etc.
	int b = 1 + ((m - 1) / 64);

	//TODO: write the actual code!
	for(int i=0; i < n; i++) {
		if (i == 0)
			knn(i, 0) = 4;
		else
			knn(i, 0) = 0;
	}
	// end TODO
	
	// FIXME: see whether C++ exception throwing is feasible, else keep using error codes
	return 0;
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
