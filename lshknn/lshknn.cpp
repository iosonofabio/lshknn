#include <pybind11/pybind11.h>

int knn_from_signature(int i, int j) {
    return i + j;
}

namespace py = pybind11;

PYBIND11_MODULE(_lshknn, m) {
    m.def("knn_from_signature", &knn_from_signature, R"pbdoc(
        Add two numbers
        Some other explanation about the add function.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
