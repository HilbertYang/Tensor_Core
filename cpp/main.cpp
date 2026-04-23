#include "tensor.h"
#include <iostream>
#include <cassert>

void test_basic() {
    std::cout << "=== Basic Operations ===\n";

    auto a = Tensor::arange(1, 7).reshape({2, 3});
    auto b = Tensor::ones({2, 3}) * 2.0f;
    std::cout << "a:\n" << a << "\n";
    std::cout << "a + b:\n" << (a + b) << "\n";
    std::cout << "a * 3:\n" << (a * 3.0f) << "\n";
}

void test_matmul() {
    std::cout << "=== Matrix Multiplication ===\n";

    // 2x3 @ 3x2
    auto a = Tensor({2, 3}, {1,2,3, 4,5,6});
    auto b = Tensor({3, 2}, {7,8, 9,10, 11,12});
    auto c = a.matmul(b);
    std::cout << "A (2x3):\n" << a << "\n";
    std::cout << "B (3x2):\n" << b << "\n";
    std::cout << "A @ B (2x2):\n" << c << "\n";

    // identity check
    auto eye = Tensor::eye(3);
    auto x = Tensor::randn({3, 4});
    assert((eye.matmul(x)).allclose(x));
    std::cout << "Identity matmul: OK\n";
}

void test_activation() {
    std::cout << "\n=== Activations ===\n";

    auto x = Tensor({1, 6}, {-2, -1, 0, 1, 2, 3});
    std::cout << "x:\n"         << x          << "\n";
    std::cout << "relu(x):\n"   << x.relu()   << "\n";
    std::cout << "sigmoid(x):\n"<< x.sigmoid()<< "\n";
    std::cout << "softmax(x):\n"<< x.softmax(1) << "\n";
}

void test_reduction() {
    std::cout << "\n=== Reductions ===\n";

    auto a = Tensor({2, 3}, {1,2,3, 4,5,6});
    std::cout << "a:\n"         << a                  << "\n";
    std::cout << "sum all: "    << a.sum().item()      << "\n";
    std::cout << "mean all: "   << a.mean().item()     << "\n";
    std::cout << "sum(dim=0):\n"<< a.sum(0)            << "\n";
    std::cout << "sum(dim=1):\n"<< a.sum(1)            << "\n";
    std::cout << "argmax(dim=1):\n" << a.argmax(1)     << "\n";
}

void test_broadcast() {
    std::cout << "\n=== Broadcasting ===\n";

    auto a = Tensor({3, 1}, {1, 2, 3});
    auto b = Tensor({1, 4}, {10, 20, 30, 40});
    std::cout << "a (3x1) + b (1x4) -> (3x4):\n" << (a + b) << "\n";
}

void test_transpose() {
    std::cout << "\n=== Transpose ===\n";

    auto a = Tensor({2, 3}, {1,2,3, 4,5,6});
    std::cout << "a:\n"            << a              << "\n";
    std::cout << "a.T:\n"          << a.transpose()  << "\n";
    std::cout << "a.T @ a (3x3):\n"<< a.transpose().matmul(a) << "\n";
}

void demo_linear_layer() {
    std::cout << "\n=== Demo: Linear Layer (Y = X @ W^T + b) ===\n";

    int batch = 4, in_features = 3, out_features = 2;
    auto X = Tensor::randn({batch, in_features});
    auto W = Tensor::randn({out_features, in_features});
    auto b = Tensor::zeros({1, out_features});

    auto Y = X.matmul(W.transpose()) + b;
    auto out = Y.relu();

    std::cout << "Input  shape: " << X.shape_str()   << "\n";
    std::cout << "Weight shape: " << W.shape_str()   << "\n";
    std::cout << "Output shape: " << out.shape_str() << "\n";
    std::cout << "Output:\n" << out << "\n";
}

int main() {
    test_basic();
    test_matmul();
    test_activation();
    test_reduction();
    test_broadcast();
    test_transpose();
    demo_linear_layer();

    std::cout << "\nAll tests passed!\n";
    return 0;
}
