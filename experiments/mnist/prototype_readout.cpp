// Where does the accuracy go: into what the neurons learned, or into the
// spiking readout? Reads the learned receptive fields from a run's weights
// PNG and classifies test images by nearest prototype (cosine similarity, no
// spikes). Cosine is scale-free, so the PNG's per-neuron scaling does not
// matter. Also runs k-means with the same number of prototypes on raw
// pixels: what an ideal unsupervised learner of that size achieves.
//
// Usage: prototype_readout WEIGHTS.png NEURONS [--data DIR] [--kmeans]
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using Img = std::vector<float>;

static uint32_t be32(std::ifstream& f) {
    unsigned char b[4];
    f.read((char*)b, 4);
    return (uint32_t)b[0] << 24 | (uint32_t)b[1] << 16 | (uint32_t)b[2] << 8 | b[3];
}

static void load(const std::string& ip, const std::string& lp, std::vector<Img>& X, std::vector<int>& y) {
    std::ifstream fi(ip, std::ios::binary), fl(lp, std::ios::binary);
    if (!fi || !fl) { std::fprintf(stderr, "cannot open MNIST in data/\n"); std::exit(1); }
    be32(fi); uint32_t n = be32(fi); be32(fi); be32(fi);
    be32(fl); be32(fl);
    X.assign(n, Img(784));
    y.resize(n);
    std::vector<unsigned char> buf(784);
    for (uint32_t i = 0; i < n; ++i) {
        fi.read((char*)buf.data(), 784);
        for (int k = 0; k < 784; ++k) X[i][k] = buf[k] / 255.0f;
        char c; fl.read(&c, 1); y[i] = (unsigned char)c;
    }
}

// Reads the grey PNG written by mnist_stdp (stored deflate blocks only).
static std::vector<uint8_t> read_png(const std::string& path, int& w, int& h) {
    std::ifstream f(path, std::ios::binary);
    std::vector<unsigned char> d((std::istreambuf_iterator<char>(f)), {});
    size_t p = 8;
    std::vector<unsigned char> z;
    while (p + 8 <= d.size()) {
        uint32_t len = (uint32_t)d[p] << 24 | d[p + 1] << 16 | d[p + 2] << 8 | d[p + 3];
        std::string type(d.begin() + p + 4, d.begin() + p + 8);
        if (type == "IHDR") {
            w = (int)((uint32_t)d[p + 8] << 24 | d[p + 9] << 16 | d[p + 10] << 8 | d[p + 11]);
            h = (int)((uint32_t)d[p + 12] << 24 | d[p + 13] << 16 | d[p + 14] << 8 | d[p + 15]);
        }
        if (type == "IDAT") z.insert(z.end(), d.begin() + p + 8, d.begin() + p + 8 + len);
        p += 12 + len;
    }
    std::vector<unsigned char> raw;
    size_t q = 2;
    for (;;) {
        int final_block = z[q] & 1;
        size_t len = z[q + 1] | z[q + 2] << 8;
        raw.insert(raw.end(), z.begin() + q + 5, z.begin() + q + 5 + len);
        q += 5 + len;
        if (final_block) break;
    }
    std::vector<uint8_t> px((size_t)w * h);
    for (int y = 0; y < h; ++y) std::copy(raw.begin() + (size_t)y * (w + 1) + 1, raw.begin() + (size_t)(y + 1) * (w + 1), px.begin() + (size_t)y * w);
    return px;
}

static void unit(Img& v) {
    double s = 0;
    for (float x : v) s += x * x;
    float n = (float)std::sqrt(s);
    if (n > 0) for (float& x : v) x /= n;
}

static int nearest(const std::vector<Img>& P, const Img& x) {
    int best = 0;
    float bs = -1e30f;
    for (size_t k = 0; k < P.size(); ++k) {
        float s = 0;
        for (int i = 0; i < 784; ++i) s += P[k][i] * x[i];
        if (s > bs) { bs = s; best = (int)k; }
    }
    return best;
}

// Labels each prototype by majority vote of the labelling images it wins,
// then scores test images by the label of their nearest prototype.
static double evaluate(const std::vector<Img>& P, const std::vector<Img>& Xl, const std::vector<int>& yl,
                       const std::vector<Img>& Xt, const std::vector<int>& yt, int from) {
    std::vector<std::vector<int>> votes(P.size(), std::vector<int>(10, 0));
    for (size_t i = 0; i < Xl.size(); ++i) votes[nearest(P, Xl[i])][yl[i]]++;
    std::vector<int> lab(P.size(), 0);
    for (size_t k = 0; k < P.size(); ++k) lab[k] = (int)(std::max_element(votes[k].begin(), votes[k].end()) - votes[k].begin());
    int ok = 0, n = 0;
    for (size_t i = from; i < Xt.size(); ++i, ++n) ok += lab[nearest(P, Xt[i])] == yt[i];
    return (double)ok / n;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: prototype_readout WEIGHTS.png NEURONS [--data DIR] [--kmeans]\n"); return 2; }
    std::string png = argv[1], data = "data";
    int neurons = std::atoi(argv[2]);
    bool kmeans = false;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--data" && i + 1 < argc) data = argv[++i];
        else if (a == "--kmeans") kmeans = true;
    }
    std::vector<Img> Xtr, Xte;
    std::vector<int> ytr, yte;
    load(data + "/train-images-idx3-ubyte", data + "/train-labels-idx1-ubyte", Xtr, ytr);
    load(data + "/t10k-images-idx3-ubyte", data + "/t10k-labels-idx1-ubyte", Xte, yte);
    // Same labelling budget as the SNN readout: 10,000 training images.
    std::vector<Img> Xl(Xtr.begin(), Xtr.begin() + 10000);
    std::vector<int> yl(ytr.begin(), ytr.begin() + 10000);
    std::vector<Img> Xlu = Xl, Xteu = Xte;
    for (auto& v : Xlu) unit(v);
    for (auto& v : Xteu) unit(v);

    int w = 0, h = 0;
    auto px = read_png(png, w, h);
    int side = (int)std::ceil(std::sqrt((double)neurons));
    std::vector<Img> P(neurons, Img(784));
    for (int k = 0; k < neurons; ++k) {
        int ox = (k % side) * 29 + 1, oy = (k / side) * 29 + 1;
        for (int i = 0; i < 784; ++i) P[k][i] = px[(size_t)(oy + i / 28) * w + ox + i % 28] / 255.0f;
        unit(P[k]);
    }
    std::printf("learned prototypes, nearest-prototype readout: %.2f%% (test 1000-9999)\n",
                100 * evaluate(P, Xlu, yl, Xteu, yte, 1000));

    if (kmeans) {
        // Spherical k-means on the whole training set, same number of prototypes.
        std::vector<Img> Xu = Xtr;
        for (auto& v : Xu) unit(v);
        std::mt19937 rng(1);
        std::vector<Img> C(neurons);
        for (int k = 0; k < neurons; ++k) C[k] = Xu[rng() % Xu.size()];
        std::vector<int> a(Xu.size());
        for (int it = 0; it < 15; ++it) {
            for (size_t i = 0; i < Xu.size(); ++i) a[i] = nearest(C, Xu[i]);
            std::vector<Img> S(neurons, Img(784, 0.0f));
            for (size_t i = 0; i < Xu.size(); ++i) for (int j = 0; j < 784; ++j) S[a[i]][j] += Xu[i][j];
            for (int k = 0; k < neurons; ++k) { C[k] = S[k]; unit(C[k]); }
        }
        std::printf("k-means, %d prototypes, nearest-prototype readout: %.2f%% (test 1000-9999)\n", neurons,
                    100 * evaluate(C, Xlu, yl, Xteu, yte, 1000));
    }
    return 0;
}
