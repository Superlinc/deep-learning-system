#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

void matrix_multiply(const float* A, const float* B, float* C,
                    int m, int n, int p) {
    for (int i = 0; i < m; ++i) {
        for (int k = 0; k < p; ++k) {
            C[i * p + k] = 0;
            for (int j = 0; j < n; ++j) {
                C[i * p + k] += A[i * n + j] * B[j * p + k];
            }
        }
    }
}

void matrix_multiply_(float* A, float x, int m, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i * n + j] *= x;
        }
    }
}

void matrix_add(float *A, const float *B, int m, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i * n + j] += B[i * n + j];
        }
    }
}

void matrix_transpose(const float* A, float* B, int m, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            B[j * m + i] = A[i * n + j];
        }
    }
}

void matrix_exp(float* A, int n) {
    for (int i = 0; i < n; ++i) {
        A[i] = exp(A[i]);
    }
}

void softmax(float* A, int m, int n) {
    for (int i = 0; i < m; ++i) {
        float sum = 0;
        for (int j = 0; j < n; ++j) {
            sum += exp(A[i * n + j]);
        }
        for (int j = 0; j < n; ++j) {
            A[i * n + j] = exp(A[i * n + j]) / sum;
        }
    }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    for (int i = 0; i < m; i += batch) {
        const float *X_batch = X + i*n;
        const unsigned char *y_batch = y + i;

        float *Z = new float[batch * k];
        matrix_multiply(X_batch, theta, Z, batch, n, k);
        softmax(Z, batch, k);

        float *I = new float[batch * k]{0};
        for (int j = 0; j < batch; ++j) {
            I[j * k + int(y_batch[j])] = -1.0f;
        }

        matrix_add(Z, I, batch, k);
        float *X_batch_transpose = new float[n * batch];
        matrix_transpose(X_batch, X_batch_transpose, batch, n);
        float *grad = new float[n * k];
        matrix_multiply(X_batch_transpose, Z, grad, n, batch, k);
        matrix_multiply_(grad, -lr/batch, n, k);
        matrix_add(theta, grad, n, k);

        delete[] Z;
        delete[] I;
        delete[] X_batch_transpose;
        delete[] grad;
    }

    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
