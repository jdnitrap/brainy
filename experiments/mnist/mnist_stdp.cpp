// Unsupervised STDP learning of MNIST digits with brainy's SNN, after
// Diehl & Cook (2015), "Unsupervised learning of digit recognition using
// spike-timing-dependent plasticity".
//
//   784 Poisson inputs --(plastic, STDP)--> N excitatory --1:1--> N inhibitory
//                                              ^                      |
//                                              +-- inhibits all others+
//
// Training never sees a label. Afterwards each excitatory neuron is labelled
// with the digit it answered most on a labelling set, and a test image is
// classified by the labels of the neurons that fire for it.
//
// Usage: mnist_stdp [--neurons N] [--train K] [--label L] [--test M]
//                   [--epochs E] [--no-learning] [--seed S] [--data DIR] [--out PREFIX]
#include "network_unified.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------- data ----

struct Dataset {
    int rows = 0, cols = 0;
    std::vector<std::vector<uint8_t>> images;
    std::vector<int> labels;
};

static uint32_t be32(std::ifstream& f) {
    unsigned char b[4];
    f.read((char*)b, 4);
    return (uint32_t)b[0] << 24 | (uint32_t)b[1] << 16 | (uint32_t)b[2] << 8 | b[3];
}

static Dataset load(const std::string& images, const std::string& labels) {
    Dataset d;
    std::ifstream fi(images, std::ios::binary), fl(labels, std::ios::binary);
    if (!fi || !fl) { std::fprintf(stderr, "cannot open %s / %s (run ./fetch_mnist.sh)\n", images.c_str(), labels.c_str()); std::exit(1); }
    if (be32(fi) != 2051 || be32(fl) != 2049) { std::fprintf(stderr, "bad MNIST header\n"); std::exit(1); }
    uint32_t n = be32(fi);
    be32(fl);
    d.rows = (int)be32(fi);
    d.cols = (int)be32(fi);
    d.images.assign(n, std::vector<uint8_t>((size_t)d.rows * d.cols));
    d.labels.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        fi.read((char*)d.images[i].data(), d.images[i].size());
        char c;
        fl.read(&c, 1);
        d.labels[i] = (unsigned char)c;
    }
    return d;
}

// ------------------------------------------------------ PNG (no zlib) ----

static uint32_t crc32(const unsigned char* p, size_t n, uint32_t c = 0xffffffffu) {
    for (size_t i = 0; i < n; ++i) {
        c ^= p[i];
        for (int k = 0; k < 8; ++k) c = (c >> 1) ^ (0xedb88320u & (0u - (c & 1u)));
    }
    return c;
}

// Grey 8-bit PNG using stored (uncompressed) deflate blocks.
static void write_png(const std::string& path, int w, int h, const std::vector<uint8_t>& px) {
    std::vector<unsigned char> raw;
    for (int y = 0; y < h; ++y) {
        raw.push_back(0);
        raw.insert(raw.end(), px.begin() + (size_t)y * w, px.begin() + (size_t)(y + 1) * w);
    }
    std::vector<unsigned char> z = {0x78, 0x01};
    uint32_t a = 1, b = 0;
    for (unsigned char c : raw) { a = (a + c) % 65521; b = (b + a) % 65521; }
    for (size_t i = 0; i < raw.size(); i += 65535) {
        size_t len = std::min<size_t>(65535, raw.size() - i);
        z.push_back(i + len == raw.size());
        z.push_back(len & 255); z.push_back(len >> 8);
        z.push_back(~len & 255); z.push_back((~len >> 8) & 255);
        z.insert(z.end(), raw.begin() + i, raw.begin() + i + len);
    }
    uint32_t adler = b << 16 | a;
    for (int s = 24; s >= 0; s -= 8) z.push_back((adler >> s) & 255);

    std::ofstream f(path, std::ios::binary);
    auto chunk = [&f](const char* type, const std::vector<unsigned char>& data) {
        unsigned char len[4] = {(unsigned char)(data.size() >> 24), (unsigned char)(data.size() >> 16),
                                (unsigned char)(data.size() >> 8), (unsigned char)data.size()};
        f.write((char*)len, 4);
        std::vector<unsigned char> td(type, type + 4);
        td.insert(td.end(), data.begin(), data.end());
        f.write((char*)td.data(), td.size());
        uint32_t c = crc32(td.data(), td.size()) ^ 0xffffffffu;
        unsigned char cb[4] = {(unsigned char)(c >> 24), (unsigned char)(c >> 16), (unsigned char)(c >> 8), (unsigned char)c};
        f.write((char*)cb, 4);
    };
    const unsigned char sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    f.write((const char*)sig, 8);
    std::vector<unsigned char> ihdr = {(unsigned char)(w >> 24), (unsigned char)(w >> 16), (unsigned char)(w >> 8), (unsigned char)w,
                                       (unsigned char)(h >> 24), (unsigned char)(h >> 16), (unsigned char)(h >> 8), (unsigned char)h,
                                       8, 0, 0, 0, 0};
    chunk("IHDR", ihdr);
    chunk("IDAT", z);
    chunk("IEND", {});
}

// --------------------------------------------------------- the network ----

struct Params {
    int neurons = 100;
    int train = 10000, label = 10000, test = 10000, epochs = 1;
    bool learning = true;
    uint32_t seed = 1;
    std::string data = "data", out = "results/run";
    // encoding and timing (1 step = 1 ms)
    int present_ms = 350;
    float max_rate = 0.06375f;    // 63.75 Hz for a white pixel
    int min_spikes = 5;           // fewer: show the image again, brighter
    // network
    float w_total = 78.0f;        // sum of input weights per excitatory neuron
    float w_inh = 1.0f;           // inhibitory -> other excitatory
    float gain = 100.0f;          // current of a weight-1 synapse (config().syn_gain)
    float theta_plus = 0.002f;    // adaptive threshold step per spike
    float a_plus = 0.01f, a_minus = 0.003f;  // STDP
};

class DigitNet {
public:
    static constexpr int kIn = 784;
    Params p;
    UnifiedNetwork net;
    std::mt19937 rng;
    int n_exc() const { return p.neurons; }
    int exc(int k) const { return kIn + k; }
    int inh(int k) const { return kIn + p.neurons + k; }

    explicit DigitNet(const Params& prm) : p(prm), net(kIn + 2 * prm.neurons), rng(prm.seed) {
        SnnConfig& c = net.config();
        c.homeostasis = false;
        c.adaptive_threshold = true;
        c.learning = p.learning;
        c.a_plus = p.a_plus;
        c.a_minus = p.a_minus;
        c.theta_plus = p.theta_plus;
        c.tau_theta = 1e7f;
        c.syn_gain = p.gain;
        for (int i = 0; i < kIn; ++i) net.addNeuron(i, NeuronType::REGULAR_SPIKING);
        for (int k = 0; k < p.neurons; ++k) {
            net.addNeuron(exc(k), NeuronType::REGULAR_SPIKING);
            net.addNeuron(inh(k), NeuronType::FAST_SPIKING);
        }
        std::uniform_real_distribution<float> u(0.0f, 0.3f);
        for (int k = 0; k < p.neurons; ++k) {
            for (int i = 0; i < kIn; ++i) net.connect(i, exc(k), u(rng));
            net.normalizeIncoming(exc(k), p.w_total);
            net.connect(exc(k), inh(k), 1.0f);
            net.setPlastic(exc(k), inh(k), false);
            for (int j = 0; j < p.neurons; ++j)
                if (j != k) net.connect(inh(k), exc(j), p.w_inh);
        }
    }

    // Presents one image; returns spike counts of the excitatory neurons.
    std::vector<int> present(const std::vector<uint8_t>& img, bool learn) {
        net.enableLearning(learn);
        net.config().theta_plus = learn ? p.theta_plus : 0.0f;  // thresholds are frozen outside training
        std::vector<int> counts(p.neurons, 0);
        float boost = 1.0f;
        std::uniform_real_distribution<float> u01(0.0f, 1.0f);
        for (int attempt = 0; attempt < 5; ++attempt) {
            std::fill(counts.begin(), counts.end(), 0);
            net.reset();
            int total = 0;
            for (int t = 0; t < p.present_ms; ++t) {
                for (int i = 0; i < kIn; ++i)
                    if (img[i] && u01(rng) < img[i] / 255.0f * p.max_rate * boost) net.forceSpike(i);
                net.step();
                for (int k = 0; k < p.neurons; ++k)
                    if (net.getNeuronSpiked(exc(k))) { ++counts[k]; ++total; }
            }
            if (total >= p.min_spikes) break;
            boost += 0.5f;
        }
        if (learn)
            for (int k = 0; k < p.neurons; ++k) net.normalizeIncoming(exc(k), p.w_total);
        return counts;
    }

    void save_weights_png(const std::string& path) {
        int side = (int)std::ceil(std::sqrt((double)p.neurons));
        int W = side * 29 + 1;
        std::vector<uint8_t> px((size_t)W * W, 40);
        for (int k = 0; k < p.neurons; ++k) {
            float mx = 1e-6f;
            for (int i = 0; i < kIn; ++i) mx = std::max(mx, net.getSynapseWeight(i, exc(k)));
            int ox = (k % side) * 29 + 1, oy = (k / side) * 29 + 1;
            for (int i = 0; i < kIn; ++i)
                px[(size_t)(oy + i / 28) * W + ox + i % 28] = (uint8_t)(255.0f * net.getSynapseWeight(i, exc(k)) / mx);
        }
        write_png(path, W, W, px);
    }
};

// ---------------------------------------------------------- baselines ----

static int nearest_centroid(const std::vector<std::vector<float>>& cent, const std::vector<uint8_t>& img) {
    int best = 0;
    double bd = 1e300;
    for (int c = 0; c < 10; ++c) {
        double d = 0;
        for (int i = 0; i < 784; ++i) { double x = img[i] / 255.0 - cent[c][i]; d += x * x; }
        if (d < bd) { bd = d; best = c; }
    }
    return best;
}

int main(int argc, char** argv) {
    Params p;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--neurons") p.neurons = std::stoi(next());
        else if (a == "--train") p.train = std::stoi(next());
        else if (a == "--label") p.label = std::stoi(next());
        else if (a == "--test") p.test = std::stoi(next());
        else if (a == "--epochs") p.epochs = std::stoi(next());
        else if (a == "--no-learning") p.learning = false;
        else if (a == "--seed") p.seed = (uint32_t)std::stoul(next());
        else if (a == "--data") p.data = next();
        else if (a == "--out") p.out = next();
        else if (a == "--w-total") p.w_total = std::stof(next());
        else if (a == "--w-inh") p.w_inh = std::stof(next());
        else if (a == "--gain") p.gain = std::stof(next());
        else if (a == "--a-plus") p.a_plus = std::stof(next());
        else if (a == "--a-minus") p.a_minus = std::stof(next());
        else if (a == "--theta-plus") p.theta_plus = std::stof(next());
        else if (a == "--rate") p.max_rate = std::stof(next()) / 1000.0f;
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    Dataset tr = load(p.data + "/train-images-idx3-ubyte", p.data + "/train-labels-idx1-ubyte");
    Dataset te = load(p.data + "/t10k-images-idx3-ubyte", p.data + "/t10k-labels-idx1-ubyte");
    p.train = std::min<int>(p.train, (int)tr.images.size());
    p.label = std::min<int>(p.label, (int)tr.images.size());
    p.test = std::min<int>(p.test, (int)te.images.size());

    std::printf("MNIST: %zu train, %zu test. Network: 784 inputs, %d excitatory, %d inhibitory, learning %s\n",
                tr.images.size(), te.images.size(), p.neurons, p.neurons, p.learning ? "on" : "off");
    DigitNet dn(p);
    std::printf("synapses: %d\n", dn.net.getNumConnections());
    auto t0 = std::chrono::steady_clock::now();
    auto secs = [&t0]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };

    // Training order: shuffled indices into the training set.
    std::vector<int> order(tr.images.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = (int)i;
    std::mt19937 shuf(p.seed + 99);
    std::shuffle(order.begin(), order.end(), shuf);

    // 1. Unsupervised training (no labels used).
    FILE* log = std::fopen((p.out + "_train.csv").c_str(), "w");
    std::fprintf(log, "seen,seconds,mean_spikes,mean_neurons_per_image,active_neurons,mean_theta\n");
    long seen = 0;
    double spike_sum = 0, distinct_sum = 0;
    std::vector<char> ever(p.neurons, 0);
    if (p.learning) {
        for (int e = 0; e < p.epochs; ++e) {
            for (int s = 0; s < p.train; ++s) {
                auto c = dn.present(tr.images[order[s]], true);
                ++seen;
                for (int k = 0; k < p.neurons; ++k) { spike_sum += c[k]; distinct_sum += c[k] > 0; if (c[k]) ever[k] = 1; }
                if (seen % 500 == 0) {
                    int act = 0;
                    double th = 0;
                    for (int k = 0; k < p.neurons; ++k) { act += ever[k]; th += dn.net.getThreshold(dn.exc(k)); }
                    std::fprintf(log, "%ld,%.1f,%.2f,%.2f,%d,%.4f\n", seen, secs(), spike_sum / 500,
                                 distinct_sum / 500, act, th / p.neurons);
                    std::fflush(log);
                    std::printf("  trained %6ld  %6.0fs  %.1f spikes/image from %.1f neurons  %d/%d neurons used  theta %.3f\n",
                                seen, secs(), spike_sum / 500, distinct_sum / 500, act, p.neurons, th / p.neurons);
                    std::fflush(stdout);
                    spike_sum = distinct_sum = 0;
                    std::fill(ever.begin(), ever.end(), 0);
                }
            }
        }
    }
    std::fclose(log);
    dn.save_weights_png(p.out + "_weights.png");
    double train_secs = secs();

    // 2. Labelling: which digit does each neuron answer to? (labels used here only)
    std::vector<std::vector<double>> resp(p.neurons, std::vector<double>(10, 0.0));
    std::vector<int> per_class(10, 0);
    for (int s = 0; s < p.label; ++s) {
        int idx = order[(size_t)(p.train + s) % order.size()];
        if (p.label >= (int)order.size()) idx = s;
        auto c = dn.present(tr.images[idx], false);
        int y = tr.labels[idx];
        ++per_class[y];
        for (int k = 0; k < p.neurons; ++k) resp[k][y] += c[k];
    }
    std::vector<int> assign(p.neurons, -1);
    for (int k = 0; k < p.neurons; ++k) {
        double best = 0;
        for (int y = 0; y < 10; ++y) {
            double r = per_class[y] ? resp[k][y] / per_class[y] : 0;
            if (r > best) { best = r; assign[k] = y; }
        }
    }
    std::vector<int> n_assigned(10, 0);
    int silent = 0;
    for (int k = 0; k < p.neurons; ++k) { if (assign[k] < 0) ++silent; else ++n_assigned[assign[k]]; }

    // 3. Test.
    int correct = 0, no_answer = 0;
    std::vector<std::vector<int>> confusion(10, std::vector<int>(10, 0));
    double test_spikes = 0;
    FILE* pred = std::fopen((p.out + "_test_predictions.csv").c_str(), "w");
    std::fprintf(pred, "index,label,predicted,spikes\n");
    for (int s = 0; s < p.test; ++s) {
        auto c = dn.present(te.images[s], false);
        double score[10] = {0};
        int tot = 0;
        for (int k = 0; k < p.neurons; ++k) {
            tot += c[k];
            if (assign[k] >= 0) score[assign[k]] += c[k];
        }
        test_spikes += tot;
        int guess = -1;
        double best = 0;
        for (int y = 0; y < 10; ++y) {
            double sc = n_assigned[y] ? score[y] / n_assigned[y] : 0;
            if (sc > best) { best = sc; guess = y; }
        }
        if (guess < 0) { ++no_answer; guess = 0; }
        confusion[te.labels[s]][guess]++;
        correct += guess == te.labels[s];
        std::fprintf(pred, "%d,%d,%d,%d\n", s, te.labels[s], guess, tot);
    }
    std::fclose(pred);
    double acc = (double)correct / p.test;

    // 4. Baselines on the same images: nearest class mean of raw pixels,
    //    fitted on the same labelling set the SNN used for its readout.
    std::vector<std::vector<float>> cent(10, std::vector<float>(784, 0.0f));
    std::vector<int> cnt(10, 0);
    for (int s = 0; s < p.label; ++s) {
        int idx = order[(size_t)(p.train + s) % order.size()];
        if (p.label >= (int)order.size()) idx = s;
        ++cnt[tr.labels[idx]];
        for (int i = 0; i < 784; ++i) cent[tr.labels[idx]][i] += tr.images[idx][i] / 255.0f;
    }
    for (int y = 0; y < 10; ++y) for (int i = 0; i < 784; ++i) cent[y][i] /= std::max(1, cnt[y]);
    int nc_correct = 0;
    for (int s = 0; s < p.test; ++s) nc_correct += nearest_centroid(cent, te.images[s]) == te.labels[s];

    // 5. Report.
    FILE* js = std::fopen((p.out + ".json").c_str(), "w");
    std::fprintf(js, "{\n  \"neurons\": %d,\n  \"learning\": %s,\n  \"train_images\": %ld,\n  \"label_images\": %d,\n"
                     "  \"test_images\": %d,\n  \"seed\": %u,\n  \"accuracy\": %.4f,\n  \"no_answer\": %d,\n"
                     "  \"nearest_centroid_accuracy\": %.4f,\n  \"silent_neurons\": %d,\n  \"mean_test_spikes\": %.2f,\n"
                     "  \"train_seconds\": %.1f,\n  \"total_seconds\": %.1f,\n  \"neurons_per_digit\": [",
                 p.neurons, p.learning ? "true" : "false", seen, p.label, p.test, p.seed, acc, no_answer,
                 (double)nc_correct / p.test, silent, test_spikes / p.test, train_secs, secs());
    for (int y = 0; y < 10; ++y) std::fprintf(js, "%d%s", n_assigned[y], y < 9 ? ", " : "]");
    std::fprintf(js, ",\n  \"confusion\": [\n");
    for (int y = 0; y < 10; ++y) {
        std::fprintf(js, "    [");
        for (int g = 0; g < 10; ++g) std::fprintf(js, "%d%s", confusion[y][g], g < 9 ? ", " : "]");
        std::fprintf(js, "%s\n", y < 9 ? "," : "");
    }
    std::fprintf(js, "  ]\n}\n");
    std::fclose(js);

    std::printf("\naccuracy %.2f%% on %d test images (%d with no spikes)\n", 100 * acc, p.test, no_answer);
    std::printf("nearest-centroid baseline %.2f%%\n", 100.0 * nc_correct / p.test);
    std::printf("neurons per digit:");
    for (int y = 0; y < 10; ++y) std::printf(" %d:%d", y, n_assigned[y]);
    std::printf("  (silent %d)\n", silent);
    std::printf("time %.0f s (training %.0f s)\n", secs(), train_secs);
    return 0;
}
