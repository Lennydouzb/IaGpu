#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Matrix.hpp"

namespace py = pybind11;

PYBIND11_MODULE(matrix_lib, m) {
    // 1. Les énumérations (Attention aux majuscules exactes du C++)
    py::enum_<e_types>(m, "e_types")
        .value("NO_TYPE", NO_TYPE)
        .value("SCALE", SCALE)
        .value("TRANSLATION", TRANSLATION)
        .value("ROTATION", ROTATION)
        .value("EMPTY", EMPTY)
        .export_values();

    // 2. La classe Matrix<float>
    py::class_<Matrix<float>>(m, "MatrixFloat")
        .def(py::init<int>())
        .def(py::init<int, int, int, std::vector<float>, int, int>(), 
             py::arg("rows"), py::arg("cols"), py::arg("type"), py::arg("list"), py::arg("firstAxis") = 0, py::arg("secondAxis") = 0)
        
        // Méthodes standards
        .def("print", &Matrix<float>::print)
        .def("getRows", &Matrix<float>::getRows)
        .def("getCols", &Matrix<float>::getCols)
        .def("getData", &Matrix<float>::getData)
        .def("transpose", &Matrix<float>::transpose)
        .def("det", &Matrix<float>::det)
		.def("hadamard", &Matrix<float>::hadamard)

        // 3. Méthodes surchargées (overload) ET constantes (py::const_)
        
        // Addition
        .def("add_mat", py::overload_cast<const Matrix<float>&>(&Matrix<float>::add, py::const_))
        .def("add_scalar", py::overload_cast<float>(&Matrix<float>::add, py::const_))
        
        // Soustraction (ajoutée)
        .def("sub_mat", py::overload_cast<const Matrix<float>&>(&Matrix<float>::sub, py::const_))
        .def("sub_scalar", py::overload_cast<float>(&Matrix<float>::sub, py::const_))
        
        // Multiplication
        .def("multiply_mat", py::overload_cast<const Matrix<float>&>(&Matrix<float>::multiply, py::const_))
        .def("multiply_scalar", py::overload_cast<float>(&Matrix<float>::multiply, py::const_));
}
