#pragma once
#include <vector>
#include <stdexcept>
#include <functional>
#include <string>
#include <ostream>
#include <initializer_list>
#include <cmath>

class Tensor {
public:
    std::vector<float> data;
    std::vector<int> shape;
    std::vector<int> strides;

    Tensor() = default;
    explicit Tensor(std::vector<int> shape, float fill = 0.0f);
    Tensor(std::vector<int> shape, std::vector<float> data);

    static Tensor zeros(std::vector<int> shape);
    static Tensor ones(std::vector<int> shape);
    static Tensor randn(std::vector<int> shape, float mean = 0.0f, float std = 1.0f);
    static Tensor arange(float start, float stop, float step = 1.0f);
    static Tensor eye(int n);

    int ndim() const { return (int)shape.size(); }
    int numel() const;
    std::string shape_str() const;

    float& at(std::vector<int> idx);
    float  at(std::vector<int> idx) const;
    float& operator[](int flat_idx) { return data[flat_idx]; }
    float  operator[](int flat_idx) const { return data[flat_idx]; }

    Tensor reshape(std::vector<int> new_shape) const;
    Tensor transpose(int dim0 = -2, int dim1 = -1) const;
    Tensor squeeze(int dim = -1) const;
    Tensor unsqueeze(int dim) const;
    Tensor flatten() const;
    Tensor slice(int dim, int start, int end) const;

    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    Tensor operator+(float scalar) const;
    Tensor operator-(float scalar) const;
    Tensor operator*(float scalar) const;
    Tensor operator/(float scalar) const;
    Tensor operator-() const;

    Tensor matmul(const Tensor& other) const;
    Tensor dot(const Tensor& other) const;

    Tensor apply(std::function<float(float)> fn) const;
    Tensor relu() const;
    Tensor sigmoid() const;
    Tensor tanh_() const;
    Tensor softmax(int dim = -1) const;
    Tensor log() const;
    Tensor exp() const;
    Tensor sqrt_() const;
    Tensor pow(float exponent) const;
    Tensor abs_() const;
    Tensor clamp(float min_val, float max_val) const;

    Tensor sum(int dim = -999, bool keepdim = false) const;
    Tensor mean(int dim = -999, bool keepdim = false) const;
    Tensor max(int dim = -999, bool keepdim = false) const;
    Tensor min(int dim = -999, bool keepdim = false) const;
    Tensor argmax(int dim) const;
    Tensor argmin(int dim) const;
    float  item() const;

    bool operator==(const Tensor& other) const;
    bool allclose(const Tensor& other, float atol = 1e-5f) const;

    friend std::ostream& operator<<(std::ostream& os, const Tensor& t);

private:
    void compute_strides();
    int  flat_index(const std::vector<int>& idx) const;
    std::vector<int> broadcast_shape(const std::vector<int>& a,
                                      const std::vector<int>& b) const;
    Tensor broadcast_to(const std::vector<int>& target_shape) const;
    Tensor elementwise(const Tensor& other,
                       std::function<float(float, float)> fn) const;
    Tensor reduce(int dim, std::function<float(float, float)> fn,
                  float init, bool keepdim) const;
};

// Free functions
Tensor operator+(float s, const Tensor& t);
Tensor operator*(float s, const Tensor& t);
Tensor matmul(const Tensor& a, const Tensor& b);
