// Walk-through of the unified SNN. Every number printed is measured, not
// asserted; the pass/fail checks are in tests_unified.cpp (make test).
#include "network_unified.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

static std::string raster(const std::vector<int>& times, int len) {
    std::string row(len, '.');
    for (int t : times) if (t >= 0 && t < len) row[t] = '|';
    return row;
}

static void demo_neuron_types() {
    std::printf("\n=== 1. Neuron types: 300 ms of steady input 10 (| = spike, 1 char = 3 ms) ===\n\n");
    const char* names[] = {"regular spiking", "fast spiking", "bursting", "dopaminergic"};
    for (int k = 0; k < 4; ++k) {
        UnifiedNetwork net(1);
        net.addNeuron(0, (NeuronType)k);
        std::vector<int> spikes;
        for (int t = 0; t < 300; ++t) {
            net.stimulate(0, 10.0f);
            net.step();
            if (net.getNeuronSpiked(0)) spikes.push_back(t / 3);
        }
        std::printf("  %-16s %s %3zu spikes\n", names[k], raster(spikes, 100).c_str(), spikes.size());
    }
}

static void demo_propagation() {
    std::printf("\n=== 2. A spike travels down a chain (w = 1.0, 1 ms synaptic delay) ===\n\n");
    UnifiedNetwork net(6);
    for (int i = 0; i < 6; ++i) net.addNeuron(i, NeuronType::REGULAR_SPIKING);
    for (int i = 0; i < 5; ++i) net.connect(i, i + 1, 1.0f);
    std::vector<std::vector<int>> spikes(6);
    net.forceSpike(0);
    for (int t = 0; t < 40; ++t) {
        net.step();
        for (int i = 0; i < 6; ++i) if (net.getNeuronSpiked(i)) spikes[i].push_back(t);
    }
    for (int i = 0; i < 6; ++i) std::printf("  neuron %d  %s\n", i, raster(spikes[i], 40).c_str());
    std::printf("\n  A weak synapse (w < 0.3) cannot fire its target with one spike; repeated\n"
                "  spikes add up, and learning can strengthen it (section 4).\n");
}

static void demo_inhibition() {
    std::printf("\n=== 3. Excitation vs inhibition ===\n\n");
    UnifiedNetwork net(2);
    net.addNeuron(0, NeuronType::REGULAR_SPIKING);
    net.addNeuron(1, NeuronType::FAST_SPIKING);  // inhibitory
    net.enableHomeostasis(false);
    net.connect(1, 0, 0.5f);
    for (int phase = 0; phase < 2; ++phase) {
        net.reset();
        std::vector<int> spikes;
        for (int t = 0; t < 300; ++t) {
            net.stimulate(0, 10.0f);
            if (phase == 1 && t >= 100 && t < 200 && t % 10 == 0) net.forceSpike(1);
            net.step();
            if (net.getNeuronSpiked(0)) spikes.push_back(t / 3);
        }
        std::printf("  %-26s %s\n", phase ? "interneuron on 100-200 ms" : "no inhibition", raster(spikes, 100).c_str());
    }
}

static void demo_learning() {
    std::printf("\n=== 4. STDP: learning the sequence A -> B -> C ===\n\n");
    UnifiedNetwork net(3);
    for (int i = 0; i < 3; ++i) net.addNeuron(i, NeuronType::REGULAR_SPIKING);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (i != j) net.connect(i, j, 0.1f);

    auto cue_a_only = [&net]() {
        net.enableLearning(false);
        net.reset();
        net.forceSpike(0);
        std::string who;
        for (int t = 0; t < 30; ++t) {
            net.step();
            for (int i = 1; i < 3; ++i)
                if (net.getNeuronSpiked(i)) who += std::string(i == 1 ? "B" : "C") + "@" + std::to_string(t) + "ms ";
        }
        net.enableLearning(true);
        return who.empty() ? std::string("nothing") : who;
    };

    std::printf("  Before training, A alone fires: %s\n\n", cue_a_only().c_str());
    std::printf("  Each trial makes A, B, C fire at 10, 20, 30 ms.\n\n");
    std::printf("  %5s %7s %7s %7s %7s %7s   %s\n", "trial", "A->B", "B->C", "A->C", "B->A", "C->A", "B first fires at");
    for (int trial = 0; trial < 30; ++trial) {
        net.reset();
        int b_first = -1;
        for (int t = 0; t < 40; ++t) {
            if (t == 10) net.forceSpike(0);
            if (t == 20) net.forceSpike(1);
            if (t == 30) net.forceSpike(2);
            net.step();
            if (b_first < 0 && net.getNeuronSpiked(1)) b_first = t;
        }
        if (trial % 5 == 0 || trial == 29)
            std::printf("  %5d %7.3f %7.3f %7.3f %7.3f %7.3f   %d ms\n", trial, net.getSynapseWeight(0, 1),
                        net.getSynapseWeight(1, 2), net.getSynapseWeight(0, 2), net.getSynapseWeight(1, 0),
                        net.getSynapseWeight(2, 0), b_first);
    }
    std::printf("\n  After training, A alone fires: %s\n", cue_a_only().c_str());
    std::printf("  Forward links grow, links back into A fade, and B starts firing before its\n"
                "  cue: the network has learned to predict it.\n");
}

static void demo_reward() {
    std::printf("\n=== 5. Reward: STDP marks synapses, dopamine decides ===\n\n");
    for (float dopamine : {1.0f, 0.0f, -1.0f}) {
        UnifiedNetwork net(2);
        net.addNeuron(0, NeuronType::REGULAR_SPIKING);
        net.addNeuron(1, NeuronType::REGULAR_SPIKING);
        net.enableRewardModulation(true);
        net.connect(0, 1, 0.3f);
        for (int trial = 0; trial < 20; ++trial) {
            net.reset();
            for (int t = 0; t < 100; ++t) {
                if (t == 10) net.forceSpike(0);
                if (t == 15) net.forceSpike(1);
                if (t == 60 && dopamine != 0.0f) net.reward(dopamine);
                net.step();
            }
        }
        std::printf("  pre->post pairing x20, reward %+.0f 45 ms later:  w 0.300 -> %.3f\n", dopamine,
                    net.getSynapseWeight(0, 1));
    }
}

static void demo_dendrites() {
    std::printf("\n=== 6. Phase 2: dendrites and BAC firing (input 50-250 ms) ===\n\n");
    const char* labels[] = {"apical 3 only", "soma 5 only", "soma 5 + apical 3"};
    float soma[] = {0, 5, 5}, apical[] = {3, 0, 3};
    for (int k = 0; k < 3; ++k) {
        UnifiedNetwork net(1);
        net.addNeuron(0, NeuronType::REGULAR_SPIKING);
        net.enablePhase2(true);
        std::vector<int> spikes;
        int plateau = 0;
        for (int t = 0; t < 300; ++t) {
            if (t >= 50 && t < 250) { net.stimulate(0, soma[k]); net.stimulateApical(0, apical[k]); }
            net.step();
            if (net.getNeuronSpiked(0)) spikes.push_back(t / 3);
            plateau += net.getNeuronInPlateau(0);
        }
        std::printf("  %-18s %s %2zu spikes, %2d ms Ca plateau\n", labels[k], raster(spikes, 100).c_str(),
                    spikes.size(), plateau);
    }
    std::printf("\n  Apical input alone stays below threshold. A somatic spike backpropagates\n"
                "  into the dendrite; together with apical input it starts a calcium plateau\n"
                "  that turns single spikes into bursts.\n");
}

static void demo_network() {
    std::printf("\n=== 7. 200-neuron E/I network with noisy input, rate per second ===\n\n");
    for (int p2 = 0; p2 < 2; ++p2) {
        const int n = 200, n_exc = 160;
        UnifiedNetwork net(n);
        net.enablePhase2(p2);
        for (int i = 0; i < n; ++i)
            net.addNeuron(i, i < n_exc ? NeuronType::REGULAR_SPIKING : NeuronType::FAST_SPIKING);
        uint32_t seed = 7;
        auto rnd = [&seed]() { seed = seed * 1664525u + 1013904223u; return (seed >> 8) / 16777216.0f; };
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                if (i != j && rnd() < 0.1f) net.connect(i, j, i < n_exc ? 0.1f : 0.3f);
        std::printf("  phase %d:", p2 + 1);
        auto start = std::chrono::steady_clock::now();
        for (int sec = 0; sec < 5; ++sec) {
            long spikes = 0;
            for (int t = 0; t < 1000; ++t) {
                for (int i = 0; i < n; ++i) net.stimulate(i, rnd() * 9.0f);
                net.step();
                for (int i = 0; i < n; ++i) spikes += net.getNeuronSpiked(i);
            }
            std::printf(" %5.1f Hz", spikes / (double)n);
        }
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        std::printf("   (%d synapses, %.0f ms for 5 s simulated)\n", net.getNumConnections(), ms);
    }
    std::printf("\n  Homeostasis pulls phase 1 down from its initial burst of activity.\n");
}

int main() {
    std::printf("Unified SNN demo (1 step = 1 ms)\n");
    demo_neuron_types();
    demo_propagation();
    demo_inhibition();
    demo_learning();
    demo_reward();
    demo_dendrites();
    demo_network();
    return 0;
}
