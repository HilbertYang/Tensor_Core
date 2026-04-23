#include "tensor.h"
#include <numeric>
#include <algorithm>
#include <cassert>
#include <random>
#include <sstream>
#include <iomanip>
#include <iostream>

// ── construction ─────────────────────────────────────────────────────────────

Tensor::Tensor(std::vector<int> shape, float fill)
    : shape(std::move(shape)) {
    compute_strides();
    data.assign(numel(), fill);
}

Tensor::Tensor(std::vector<int> shape, std::vector<float> data)
    : data(std::move(data)), shape(std::move(shape)) {
    compute_strides();
    if ((int)this->data.size() != numel())
        throw std::invalid_argument("data size does not match shape");
}

void Tensor::compute_strides() {
    strides.resize(shape.size());
    int s = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        strides[i] = s;
        s *= shape[i];
    }
}

int Tensor::numel() const {
    if (shape.empty()) return 1;
    int n = 1;
    for (int d : shape) n *= d;
    return n;
}

std::string Tensor::shape_str() const {
    std::string s = "(";
    for (int i = 0; i < (int)shape.size(); ++i) {
        s += std::to_string(shape[i]);
        if (i + 1 < (int)shape.size()) s += ", ";
    }
    return s + ")";
}

// ── factories ─────────────────────────────────────────────────────────────────

Tensor Tensor::zeros(std::vector<int> shape) { return Tensor(shape, 0.0f); }
Tensor Tensor::ones(std::vector<int> shape)  { return Tensor(shape, 1.0f); }

Tensor Tensor::randn(std::vector<int> shape, float mean, float std) {
    Tensor t(shape);
    static std::mt19937 gen{std::random_device{}()};
    std::normal_distribution<float> dist(mean, std);
    for (float& v : t.data) v = dist(gen);
    return t;
}

Tensor Tensor::arange(float start, float stop, float step) {
    std::vector<float> d;
    for (float v = start; v < stop; v += step) d.push_back(v);
    return Tensor({(int)d.size()}, d);
}

Tensor Tensor::eye(int n) {
    Tensor t({n, n}, 0.0f);
    for (int i = 0; i < n; ++i) t.data[i * n + i] = 1.0f;
    return t;
}

// ── indexing ──────────────────────────────────────────────────────────────────

int Tensor::flat_index(const std::vector<int>& idx) const {
    if (idx.size() != shape.size())
        throw std::out_of_range("wrong number of indices");
    int flat = 0;
    for (int i = 0; i < (int)idx.size(); ++i) {
        int d = idx[i] < 0 ? idx[i] + shape[i] : idx[i];
        if (d < 0 || d >= shape[i]) throw std::out_of_range("index out of bounds");
        flat += d * strides[i];
    }
    return flat;
}

float& Tensor::at(std::vector<int> idx)       { return data[flat_index(idx)]; }
float  Tensor::at(std::vector<int> idx) const { return data[flat_index(idx)]; }

// ── shape operations ──────────────────────────────────────────────────────────

Tensor Tensor::reshape(std::vector<int> new_shape) const {
    // allow one -1 dimension
    int neg = -1, known = 1;
    for (int i = 0; i < (int)new_shape.size(); ++i) {
        if (new_shape[i] == -1) {
            if (neg != -1) throw std::invalid_argument("only one -1 allowed in reshape");
            neg = i;
        } else {
            known *= new_shape[i];
        }
    }
    if (neg != -1) new_shape[neg] = numel() / known;
    if (known * (neg == -1 ? 1 : new_shape[neg]) != numel())
        throw std::invalid_argument("reshape: incompatible size");
    Tensor out(new_shape, data);
    return out;
}

Tensor Tensor::flatten() const { return reshape({numel()}); }

Tensor Tensor::transpose(int dim0, int dim1) const {
    int n = ndim();
    if (dim0 < 0) dim0 += n;
    if (dim1 < 0) dim1 += n;
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::swap(perm[dim0], perm[dim1]);

    std::vector<int> new_shape(n);
    for (int i = 0; i < n; ++i) new_shape[i] = shape[perm[i]];

    Tensor out(new_shape);
    std::vector<int> src_idx(n), dst_idx(n);

    std::function<void(int)> fill = [&](int d) {
        if (d == n) {
            for (int i = 0; i < n; ++i) src_idx[perm[i]] = dst_idx[i];
            out.at(dst_idx) = at(src_idx);
            return;
        }
        for (int i = 0; i < new_shape[d]; ++i) { dst_idx[d] = i; fill(d + 1); }
    };
    fill(0);
    return out;
}

Tensor Tensor::unsqueeze(int dim) const {
    int n = ndim() + 1;
    if (dim < 0) dim += n;
    std::vector<int> ns = shape;
    ns.insert(ns.begin() + dim, 1);
    return reshape(ns);
}

Tensor Tensor::squeeze(int dim) const {
    if (dim == -1) {
        std::vector<int> ns;
        for (int d : shape) if (d != 1) ns.push_back(d);
        if (ns.empty()) ns = {1};
        return reshape(ns);
    }
    if (dim < 0) dim += ndim();
    if (shape[dim] != 1) throw std::invalid_argument("squeeze: dim size is not 1");
    std::vector<int> ns = shape;
    ns.erase(ns.begin() + dim);
    return reshape(ns);
}

Tensor Tensor::slice(int dim, int start, int end) const {
    if (dim < 0) dim += ndim();
    std::vector<int> ns = shape;
    ns[dim] = end - start;
    Tensor out(ns);
    std::vector<int> idx(ndim(), 0);

    std::function<void(int)> copy = [&](int d) {
        if (d == ndim()) {
            out.at(idx) = at(idx);
            return;
        }
        int lo = (d == dim) ? start : 0;
        int hi = (d == dim) ? end   : shape[d];
        for (int i = lo; i < hi; ++i) {
            idx[d] = i;
            auto out_idx = idx;
            out_idx[dim] -= start;
            copy(d + 1);
        }
    };

    // simpler direct copy
    std::vector<int> src(ndim(), 0), dst(ndim(), 0);
    std::function<void(int)> cp = [&](int d) {
        if (d == ndim()) { out.at(dst) = at(src); return; }
        for (int i = 0; i < ns[d]; ++i) {
            dst[d] = i;
            src[d] = (d == dim) ? start + i : i;
            cp(d + 1);
        }
    };
    cp(0);
    return out;
}

// ── broadcasting ──────────────────────────────────────────────────────────────

std::vector<int> Tensor::broadcast_shape(const std::vector<int>& a,
                                          const std::vector<int>& b) const {
    int n = std::max(a.size(), b.size());
    std::vector<int> out(n);
    for (int i = 0; i < n; ++i) {
        int da = (i < (int)a.size()) ? a[a.size() - 1 - i] : 1;
        int db = (i < (int)b.size()) ? b[b.size() - 1 - i] : 1;
        if (da != db && da != 1 && db != 1)
            throw std::invalid_argument("shapes are not broadcastable");
        out[n - 1 - i] = std::max(da, db);
    }
    return out;
}

Tensor Tensor::broadcast_to(const std::vector<int>& target) const {
    int tn = target.size();
    int sn = shape.size();
    Tensor out(target);
    std::vector<int> dst(tn, 0);
    std::function<void(int)> fill = [&](int d) {
        if (d == tn) {
            std::vector<int> src(sn);
            for (int i = 0; i < sn; ++i) {
                int ti = tn - sn + i;
                src[i] = (shape[i] == 1) ? 0 : dst[ti];
            }
            out.at(dst) = at(src);
            return;
        }
        for (int i = 0; i < target[d]; ++i) { dst[d] = i; fill(d + 1); }
    };
    fill(0);
    return out;
}

Tensor Tensor::elementwise(const Tensor& other,
                            std::function<float(float, float)> fn) const {
    auto bs = broadcast_shape(shape, other.shape);
    auto a = broadcast_to(bs);
    auto b = other.broadcast_to(bs);
    Tensor out(bs);
    for (int i = 0; i < out.numel(); ++i) out.data[i] = fn(a.data[i], b.data[i]);
    return out;
}

// ── arithmetic ────────────────────────────────────────────────────────────────

Tensor Tensor::operator+(const Tensor& o) const {
    return elementwise(o, [](float a, float b){ return a + b; });
}
Tensor Tensor::operator-(const Tensor& o) const {
    return elementwise(o, [](float a, float b){ return a - b; });
}
Tensor Tensor::operator*(const Tensor& o) const {
    return elementwise(o, [](float a, float b){ return a * b; });
}
Tensor Tensor::operator/(const Tensor& o) const {
    return elementwise(o, [](float a, float b){ return a / b; });
}
Tensor Tensor::operator+(float s) const {
    return apply([s](float v){ return v + s; });
}
Tensor Tensor::operator-(float s) const {
    return apply([s](float v){ return v - s; });
}
Tensor Tensor::operator*(float s) const {
    return apply([s](float v){ return v * s; });
}
Tensor Tensor::operator/(float s) const {
    return apply([s](float v){ return v / s; });
}
Tensor Tensor::operator-() const {
    return apply([](float v){ return -v; });
}
Tensor operator+(float s, const Tensor& t) { return t + s; }
Tensor operator*(float s, const Tensor& t) { return t * s; }

// ── matrix multiplication ─────────────────────────────────────────────────────

Tensor Tensor::matmul(const Tensor& other) const {
    // supports (M,K) x (K,N) -> (M,N)  and batched (...,M,K) x (...,K,N)
    if (ndim() < 2 || other.ndim() < 2)
        throw std::invalid_argument("matmul requires at least 2-D tensors");

    int M = shape[ndim() - 2];
    int K = shape[ndim() - 1];
    int K2 = other.shape[other.ndim() - 2];
    int N = other.shape[other.ndim() - 1];
    if (K != K2)
        throw std::invalid_argument("matmul: inner dimensions must match");

    // batch dimensions
    std::vector<int> abatch(shape.begin(), shape.end() - 2);
    std::vector<int> bbatch(other.shape.begin(), other.shape.end() - 2);
    auto batch = broadcast_shape(abatch, bbatch);

    std::vector<int> out_shape = batch;
    out_shape.push_back(M);
    out_shape.push_back(N);
    Tensor out(out_shape, 0.0f);

    int batch_size = 1;
    for (int d : batch) batch_size *= d;

    // build a flat batch tensor (broadcast both operands)
    Tensor a_bc = (abatch.empty() ? *this : reshape(abatch).broadcast_to(batch).reshape(
        [&]{ auto s=batch; s.push_back(M); s.push_back(K); return s; }()));
    // simpler: just iterate
    // We iterate over batches using flat index
    auto batch_idx = [&](int flat, const std::vector<int>& bshape) -> std::vector<int> {
        std::vector<int> idx(bshape.size());
        for (int i = (int)bshape.size()-1; i >= 0; --i) {
            idx[i] = flat % bshape[i]; flat /= bshape[i];
        }
        return idx;
    };

    for (int b = 0; b < batch_size; ++b) {
        auto bidx = batch_idx(b, batch);

        // get A and B slice starting indices
        auto a_bidx = bidx;
        for (int i = 0; i < (int)abatch.size(); ++i)
            a_bidx[i] = (abatch.empty() || abatch[i] == 1) ? 0 : bidx[bidx.size()-abatch.size()+i];

        auto get_a = [&](int m, int k) -> float {
            std::vector<int> idx = a_bidx;
            idx.push_back(m); idx.push_back(k);
            // adjust for broadcast
            while ((int)idx.size() > ndim()) idx.erase(idx.begin());
            for (int i = 0; i < ndim()-2; ++i)
                if (shape[i] == 1) idx[i] = 0;
            return at(idx);
        };
        auto get_b = [&](int k, int n) -> float {
            auto bidx2 = bidx;
            std::vector<int> idx = bidx2;
            idx.push_back(k); idx.push_back(n);
            while ((int)idx.size() > other.ndim()) idx.erase(idx.begin());
            for (int i = 0; i < other.ndim()-2; ++i)
                if (other.shape[i] == 1) idx[i] = 0;
            return other.at(idx);
        };

        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) acc += get_a(m, k) * get_b(k, n);
                std::vector<int> oidx = bidx;
                oidx.push_back(m); oidx.push_back(n);
                out.at(oidx) = acc;
            }
        }
    }
    return out;
}

Tensor matmul(const Tensor& a, const Tensor& b) { return a.matmul(b); }
Tensor Tensor::dot(const Tensor& other) const    { return matmul(other); }

// ── activation / element-wise math ───────────────────────────────────────────

Tensor Tensor::apply(std::function<float(float)> fn) const {
    Tensor out(shape);
    for (int i = 0; i < numel(); ++i) out.data[i] = fn(data[i]);
    return out;
}

Tensor Tensor::relu()    const { return apply([](float v){ return v > 0 ? v : 0.0f; }); }
Tensor Tensor::sigmoid() const { return apply([](float v){ return 1.0f/(1.0f+std::exp(-v)); }); }
Tensor Tensor::tanh_()   const { return apply([](float v){ return std::tanh(v); }); }
Tensor Tensor::log()     const { return apply([](float v){ return std::log(v); }); }
Tensor Tensor::exp()     const { return apply([](float v){ return std::exp(v); }); }
Tensor Tensor::sqrt_()   const { return apply([](float v){ return std::sqrt(v); }); }
Tensor Tensor::abs_()    const { return apply([](float v){ return std::abs(v); }); }
Tensor Tensor::pow(float e) const { return apply([e](float v){ return std::pow(v, e); }); }
Tensor Tensor::clamp(float lo, float hi) const {
    return apply([lo, hi](float v){ return std::max(lo, std::min(hi, v)); });
}

Tensor Tensor::softmax(int dim) const {
    if (dim < 0) dim += ndim();
    Tensor m = max(dim, true);
    Tensor shifted = *this - m.broadcast_to(shape);
    Tensor e = shifted.exp();
    Tensor s = e.sum(dim, true);
    return e / s.broadcast_to(e.shape);
}

// ── reductions ────────────────────────────────────────────────────────────────

Tensor Tensor::reduce(int dim, std::function<float(float, float)> fn,
                       float init, bool keepdim) const {
    std::vector<int> out_shape = shape;
    out_shape[dim] = 1;
    Tensor out(out_shape, init);

    std::vector<int> idx(ndim(), 0);
    std::function<void(int)> rec = [&](int d) {
        if (d == ndim()) {
            auto oidx = idx;
            oidx[dim] = 0;
            out.at(oidx) = fn(out.at(oidx), at(idx));
            return;
        }
        for (int i = 0; i < shape[d]; ++i) { idx[d] = i; rec(d + 1); }
    };
    rec(0);

    if (!keepdim) {
        std::vector<int> sq;
        for (int i = 0; i < (int)out_shape.size(); ++i)
            if (i != dim) sq.push_back(out_shape[i]);
        if (sq.empty()) sq = {1};
        return out.reshape(sq);
    }
    return out;
}

Tensor Tensor::sum(int dim, bool keepdim) const {
    if (dim == -999) {
        float s = 0; for (float v : data) s += v;
        return Tensor({1}, {s});
    }
    if (dim < 0) dim += ndim();
    return reduce(dim, [](float a, float b){ return a + b; }, 0.0f, keepdim);
}

Tensor Tensor::mean(int dim, bool keepdim) const {
    if (dim == -999) {
        float s = 0; for (float v : data) s += v;
        return Tensor({1}, {s / numel()});
    }
    if (dim < 0) dim += ndim();
    auto s = reduce(dim, [](float a, float b){ return a + b; }, 0.0f, keepdim);
    return s / (float)shape[dim];
}

Tensor Tensor::max(int dim, bool keepdim) const {
    if (dim == -999) {
        float m = data[0]; for (float v : data) m = std::max(m, v);
        return Tensor({1}, {m});
    }
    if (dim < 0) dim += ndim();
    return reduce(dim, [](float a, float b){ return std::max(a, b); },
                  -std::numeric_limits<float>::infinity(), keepdim);
}

Tensor Tensor::min(int dim, bool keepdim) const {
    if (dim == -999) {
        float m = data[0]; for (float v : data) m = std::min(m, v);
        return Tensor({1}, {m});
    }
    if (dim < 0) dim += ndim();
    return reduce(dim, [](float a, float b){ return std::min(a, b); },
                  std::numeric_limits<float>::infinity(), keepdim);
}

Tensor Tensor::argmax(int dim) const {
    if (dim < 0) dim += ndim();
    std::vector<int> out_shape = shape;
    out_shape[dim] = 1;
    Tensor out(out_shape, 0.0f);
    Tensor maxval(out_shape, -std::numeric_limits<float>::infinity());

    std::vector<int> idx(ndim(), 0);
    std::function<void(int)> rec = [&](int d) {
        if (d == ndim()) {
            auto oidx = idx; oidx[dim] = 0;
            if (at(idx) > maxval.at(oidx)) {
                maxval.at(oidx) = at(idx);
                out.at(oidx) = (float)idx[dim];
            }
            return;
        }
        for (int i = 0; i < shape[d]; ++i) { idx[d] = i; rec(d + 1); }
    };
    rec(0);
    return out.squeeze(dim);
}

Tensor Tensor::argmin(int dim) const {
    if (dim < 0) dim += ndim();
    std::vector<int> out_shape = shape;
    out_shape[dim] = 1;
    Tensor out(out_shape, 0.0f);
    Tensor minval(out_shape, std::numeric_limits<float>::infinity());

    std::vector<int> idx(ndim(), 0);
    std::function<void(int)> rec = [&](int d) {
        if (d == ndim()) {
            auto oidx = idx; oidx[dim] = 0;
            if (at(idx) < minval.at(oidx)) {
                minval.at(oidx) = at(idx);
                out.at(oidx) = (float)idx[dim];
            }
            return;
        }
        for (int i = 0; i < shape[d]; ++i) { idx[d] = i; rec(d + 1); }
    };
    rec(0);
    return out.squeeze(dim);
}

float Tensor::item() const {
    if (numel() != 1) throw std::runtime_error("item(): tensor has more than one element");
    return data[0];
}

// ── comparison ────────────────────────────────────────────────────────────────

bool Tensor::operator==(const Tensor& o) const {
    return shape == o.shape && data == o.data;
}

bool Tensor::allclose(const Tensor& o, float atol) const {
    if (shape != o.shape) return false;
    for (int i = 0; i < numel(); ++i)
        if (std::abs(data[i] - o.data[i]) > atol) return false;
    return true;
}

// ── printing ──────────────────────────────────────────────────────────────────

static void print_recursive(std::ostream& os, const Tensor& t,
                             int dim, std::vector<int>& idx, int indent) {
    if (dim == t.ndim() - 1) {
        os << "[";
        for (int i = 0; i < t.shape[dim]; ++i) {
            idx[dim] = i;
            os << std::setw(8) << std::fixed << std::setprecision(4) << t.at(idx);
            if (i + 1 < t.shape[dim]) os << ", ";
        }
        os << "]";
        return;
    }
    os << "[\n";
    for (int i = 0; i < t.shape[dim]; ++i) {
        idx[dim] = i;
        os << std::string(indent + 2, ' ');
        print_recursive(os, t, dim + 1, idx, indent + 2);
        if (i + 1 < t.shape[dim]) os << ",";
        os << "\n";
    }
    os << std::string(indent, ' ') << "]";
}

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    os << "Tensor(shape=" << t.shape_str() << ")\n";
    if (t.ndim() == 0 || t.numel() == 0) { os << "[]"; return os; }
    std::vector<int> idx(t.ndim(), 0);
    print_recursive(os, t, 0, idx, 0);
    return os;
}
