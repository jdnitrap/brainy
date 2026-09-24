// Pass/fail checks for the unified SNN. Build and run: make test
#include "network_unified.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

static int g_fail = 0, g_pass = 0;

#define CHECK(cond, ...)                                  \
    do {                                                  \
        char msg_[256];                                   \
        std::snprintf(msg_, sizeof msg_, __VA_ARGS__);    \
        if (cond) { ++g_pass; std::printf("  PASS  %s\n", msg_); } \
        else { ++g_fail; std::printf("  FAIL  %s   [%s]\n", msg_, #cond); } \
    } while (0)

static void section(const char* name) { std::printf("%s\n", name); }

// Steady current into the soma for `steps`; returns the spike times.
static std::vector<int> drive(UnifiedNetwork& net, int id, float current, int steps) {
    std::vector<int> times;
    for (int t = 0; t < steps; ++t) {
        net.stimulate(id, current);
        net.step();
        if (net.getNeuronSpiked(id)) times.push_back(net.getTime() - 1);
    }
    return times;
}

static UnifiedNetwork quiet(int n, NeuronType type = NeuronType::REGULAR_SPIKING) {
    UnifiedNetwork net(n);
    for (int i = 0; i < n; ++i) net.addNeuron(i, type);
    net.enableHomeostasis(false);
    net.enableLearning(false);
    return net;
}

// ---------------------------------------------------------------------------

static void test_neurons() {
    section("neurons");
    const char* names[] = {"regular", "fast", "bursting", "dopaminergic"};
    NeuronType types[] = {NeuronType::REGULAR_SPIKING, NeuronType::FAST_SPIKING,
                          NeuronType::BURSTING, NeuronType::DOPAMINERGIC};
    for (int p2 = 0; p2 < 2; ++p2) {
        for (int k = 0; k < 4; ++k) {
            auto net = quiet(1, types[k]);
            net.enablePhase2(p2);
            net.enableStructuralPlasticity(false);
            float v0 = net.getNeuronVoltage(0), lo = v0, hi = v0;
            int spikes = 0;
            for (int t = 0; t < 1000; ++t) {
                net.step();
                spikes += net.getNeuronSpiked(0);
                lo = std::fmin(lo, net.getNeuronVoltage(0));
                hi = std::fmax(hi, net.getNeuronVoltage(0));
            }
            CHECK(spikes == 0 && hi - lo < 0.5f, "%s, phase %d: no input stays at rest (%.1f mV, drift %.2f)",
                  names[k], p2 + 1, v0, hi - lo);
        }
    }

    auto rs = quiet(1);
    int r5 = (int)drive(rs, 0, 5, 1000).size();
    rs = quiet(1);
    int r10 = (int)drive(rs, 0, 10, 1000).size();
    rs = quiet(1);
    int r20 = (int)drive(rs, 0, 20, 1000).size();
    CHECK(r5 >= 3 && r5 < r10 && r10 < r20 && r20 < 100,
          "regular spiking: rate rises with input (%d, %d, %d Hz at 5, 10, 20)", r5, r10, r20);

    auto fs = quiet(1, NeuronType::FAST_SPIKING);
    int f10 = (int)drive(fs, 0, 10, 1000).size();
    CHECK(f10 > 3 * r10, "fast spiking fires much faster than regular (%d vs %d Hz at 10)", f10, r10);

    auto bu = quiet(1, NeuronType::BURSTING);
    auto bt = drive(bu, 0, 10, 1000);
    int min_isi = 1000, max_isi = 0;
    for (size_t i = 1; i < bt.size(); ++i) {
        min_isi = std::min(min_isi, bt[i] - bt[i - 1]);
        max_isi = std::max(max_isi, bt[i] - bt[i - 1]);
    }
    CHECK(bt.size() > 10 && min_isi <= 8 && max_isi >= 3 * min_isi,
          "bursting: short gaps inside bursts, long gaps between (ISI %d..%d ms)", min_isi, max_isi);

    auto a = quiet(1, NeuronType::FAST_SPIKING);
    int with_ref = (int)drive(a, 0, 200, 200).size();
    auto b = quiet(1, NeuronType::FAST_SPIKING);
    b.enableRefractory(false);
    int without_ref = (int)drive(b, 0, 200, 200).size();
    CHECK(without_ref > with_ref, "refractory period caps the rate (%d vs %d spikes in 200 ms)", with_ref, without_ref);

    bool finite = true;
    auto big = quiet(1);
    for (int t = 0; t < 500; ++t) {
        big.stimulate(0, 5000.0f);
        big.step();
        finite = finite && std::isfinite(big.getNeuronVoltage(0));
    }
    CHECK(finite, "huge input never produces inf/NaN");
}

static void test_synapses() {
    section("synapses and propagation");
    for (float w : {0.1f, 0.2f, 1.0f}) {
        auto net = quiet(2);
        net.connect(0, 1, w);
        net.forceSpike(0);
        int post = 0;
        for (int t = 0; t < 50; ++t) { net.step(); post += net.getNeuronSpiked(1); }
        if (w < 0.25f) CHECK(post == 0, "one spike through a weak synapse (w=%.1f) does not fire the target", w);
        else CHECK(post == 1, "one spike through a strong synapse (w=%.1f) fires the target once", w);
    }

    {
        auto net = quiet(2);
        net.connect(0, 1, 0.2f);
        int pre = 0, post = 0;
        for (int t = 0; t < 1000; ++t) {
            net.stimulate(0, 20);
            net.step();
            pre += net.getNeuronSpiked(0);
            post += net.getNeuronSpiked(1);
        }
        CHECK(post > 0, "repeated spikes through a weak synapse add up (pre %d, post %d spikes)", pre, post);
    }

    {
        const int n = 10;
        auto net = quiet(n);
        for (int i = 0; i + 1 < n; ++i) net.connect(i, i + 1, 1.0f);
        net.forceSpike(0);
        std::vector<int> first(n, -1);
        for (int t = 0; t < 100; ++t) {
            net.step();
            for (int i = 0; i < n; ++i)
                if (first[i] < 0 && net.getNeuronSpiked(i)) first[i] = t;
        }
        bool ordered = first[n - 1] > 0;
        for (int i = 1; i < n; ++i) ordered = ordered && first[i] > first[i - 1];
        CHECK(ordered, "a spike travels down a 10-neuron chain in order (last neuron at %d ms)", first[n - 1]);
    }

    {
        auto net = quiet(2);
        net.connect(0, 1, 0.5f);
        net.forceSpike(0);
        net.step();
        float on_total = 0.0f;
        int on_steps = 0;
        for (int t = 0; t < 40; ++t) {
            float c = net.getSynapticCurrent(1);
            on_total += c;
            on_steps += c > 0.01f;
            net.step();
        }
        auto net2 = quiet(2);
        net2.enablePSC(false);
        net2.connect(0, 1, 0.5f);
        net2.forceSpike(0);
        net2.step();
        float off_total = 0.0f;
        int off_steps = 0;
        for (int t = 0; t < 40; ++t) {
            float c = net2.getSynapticCurrent(1);
            off_total += c;
            off_steps += c > 0.01f;
            net2.step();
        }
        CHECK(on_steps >= 10 && off_steps == 1 && std::fabs(on_total - off_total) < 0.02f * off_total,
              "PSC spreads a spike's current over time (%d steps vs %d), same total charge (%.1f vs %.1f)",
              on_steps, off_steps, on_total, off_total);
    }

    {
        auto base = quiet(2);
        base.addNeuron(1, NeuronType::FAST_SPIKING);
        base.connect(1, 0, 1.0f);
        int alone = 0, inhibited = 0;
        for (int t = 0; t < 1000; ++t) { base.stimulate(0, 10); base.step(); alone += base.getNeuronSpiked(0); }
        base.reset();
        for (int t = 0; t < 1000; ++t) {
            base.stimulate(0, 10);
            if (t % 10 == 0) base.forceSpike(1);
            base.step();
            inhibited += base.getNeuronSpiked(0);
        }
        CHECK(inhibited * 2 < alone, "a fast-spiking (inhibitory) neuron suppresses its target (%d -> %d Hz)",
              alone, inhibited);
    }

    {
        SnnConfig cfg;
        cfg.short_term = true;
        UnifiedSynapse dep(0, 1, 1.0f, false, Target::AUTO);
        UnifiedSynapse fac(0, 1, 1.0f, false, Target::AUTO);
        fac.U = 0.1f; fac.tau_facil = 500.0f; fac.tau_rec = 100.0f;
        float d1 = dep.onPreSpike(0, cfg), f1 = fac.onPreSpike(0, cfg), d5 = 0, f5 = 0;
        for (int k = 1; k < 5; ++k) { d5 = dep.onPreSpike(20 * k, cfg); f5 = fac.onPreSpike(20 * k, cfg); }
        CHECK(std::fabs(d1 - 1) < 1e-5f && std::fabs(f1 - 1) < 1e-5f, "short-term: first spike has efficacy 1");
        CHECK(d5 < 0.6f * d1, "short-term: depressing synapse weakens at 50 Hz (5th spike %.2f)", d5);
        CHECK(f5 > 1.5f * f1, "short-term: facilitating synapse strengthens at 50 Hz (5th spike %.2f)", f5);
        float rec = dep.onPreSpike(80 + 2000, cfg);
        CHECK(rec > 0.95f, "short-term: depressed synapse recovers after 2 s (%.2f)", rec);
    }
}

// Pre spike at 10 ms, post spike at 10 + dt ms (dt may be negative).
static float pairing_change(int dt, int repeats = 1) {
    UnifiedNetwork net(2);
    net.addNeuron(0, NeuronType::REGULAR_SPIKING);
    net.addNeuron(1, NeuronType::REGULAR_SPIKING);
    net.enableHomeostasis(false);
    net.connect(0, 1, 0.1f);  // too weak to make 1 fire by itself
    float w0 = net.getSynapseWeight(0, 1);
    for (int r = 0; r < repeats; ++r) {
        net.reset();
        for (int t = 0; t < 60; ++t) {
            if (t == 10) net.forceSpike(0);
            if (t == 10 + dt) net.forceSpike(1);
            net.step();
        }
    }
    return net.getSynapseWeight(0, 1) - w0;
}

static void test_stdp() {
    section("STDP");
    float ltp5 = pairing_change(5), ltp20 = pairing_change(20), ltd5 = pairing_change(-5), none = pairing_change(0);
    CHECK(ltp5 > 0, "pre 5 ms before post strengthens (%+.4f)", ltp5);
    CHECK(ltp20 > 0 && ltp20 < ltp5, "the effect shrinks with the gap (%+.4f at 20 ms)", ltp20);
    CHECK(ltd5 < 0, "pre 5 ms after post weakens (%+.4f)", ltd5);
    CHECK(none == 0.0f, "simultaneous spikes change nothing");

    {
        UnifiedNetwork net(2);
        net.addNeuron(0, NeuronType::REGULAR_SPIKING);
        net.addNeuron(1, NeuronType::REGULAR_SPIKING);
        net.enableHomeostasis(false);
        net.connect(0, 1, 0.1f);
        net.forceSpike(0);
        net.step();
        for (int t = 0; t < 4; ++t) net.step();
        net.forceSpike(1);
        net.step();
        float after_pair = net.getSynapseWeight(0, 1);
        net.run(1000);
        CHECK(after_pair > 0.1f && net.getSynapseWeight(0, 1) == after_pair,
              "each pairing counts once: no drift while idle (%.4f stays)", after_pair);
    }

    float many = pairing_change(5, 500);
    CHECK(0.1f + many <= 1.0f && 0.1f + many > 0.9f, "weights stay within [0, w_max] (%.3f after 500 pairings)",
          0.1f + many);

    // Sequence learning and recall: A -> B -> C.
    UnifiedNetwork net(3);
    for (int i = 0; i < 3; ++i) net.addNeuron(i, NeuronType::REGULAR_SPIKING);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (i != j) net.connect(i, j, 0.1f);
    auto recall = [&net]() {
        net.enableLearning(false);
        net.reset();
        net.forceSpike(0);
        int b = -1, c = -1;
        for (int t = 0; t < 60; ++t) {
            net.step();
            if (b < 0 && net.getNeuronSpiked(1)) b = t;
            if (c < 0 && net.getNeuronSpiked(2)) c = t;
        }
        net.enableLearning(true);
        return std::make_pair(b, c);
    };
    auto before = recall();
    CHECK(before.first < 0 && before.second < 0, "before training, A alone does not fire B or C");
    int b_last = -1;
    for (int trial = 0; trial < 50; ++trial) {
        net.reset();
        b_last = -1;
        for (int t = 0; t < 40; ++t) {
            if (t == 10) net.forceSpike(0);
            if (t == 20) net.forceSpike(1);
            if (t == 30) net.forceSpike(2);
            net.step();
            if (b_last < 0 && net.getNeuronSpiked(1)) b_last = t;
        }
    }
    float ab = net.getSynapseWeight(0, 1), bc = net.getSynapseWeight(1, 2), ac = net.getSynapseWeight(0, 2);
    float ba = net.getSynapseWeight(1, 0), ca = net.getSynapseWeight(2, 0);
    CHECK(ab > 0.5f && bc > 0.5f && ac > 0.5f, "training strengthens A->B, B->C, A->C (%.2f, %.2f, %.2f)", ab, bc,
          ac);
    CHECK(ba < 0.05f && ca < 0.05f, "links back into A weaken (B->A %.3f, C->A %.3f)", ba, ca);
    CHECK(b_last >= 0 && b_last < 20, "the network learns to anticipate: B fires at %d ms, before its cue at 20",
          b_last);
    auto after = recall();
    CHECK(after.first > 0 && after.second > 0 && after.second < 10,
          "after training, A alone brings back B and C (at %d and %d ms)", after.first, after.second);
}

static void test_reward() {
    section("reward modulation");
    auto paired = [](int dt, int reward_delay, float dopamine) {
        UnifiedNetwork net(2);
        net.addNeuron(0, NeuronType::REGULAR_SPIKING);
        net.addNeuron(1, NeuronType::REGULAR_SPIKING);
        net.enableHomeostasis(false);
        net.enableRewardModulation(true);
        net.connect(0, 1, 0.3f);
        for (int t = 0; t < 10 + std::abs(dt) + 5 + reward_delay; ++t) {
            if (t == 10) net.forceSpike(0);
            if (t == 10 + dt) net.forceSpike(1);
            if (t == 10 + std::abs(dt) + reward_delay) net.reward(dopamine);
            net.step();
        }
        return net.getSynapseWeight(0, 1) - 0.3f;
    };
    {
        UnifiedNetwork net(2);
        net.addNeuron(0, NeuronType::REGULAR_SPIKING);
        net.addNeuron(1, NeuronType::REGULAR_SPIKING);
        net.enableHomeostasis(false);
        net.enableRewardModulation(true);
        net.connect(0, 1, 0.3f);
        for (int t = 0; t < 30; ++t) {
            if (t == 10) net.forceSpike(0);
            if (t == 15) net.forceSpike(1);
            net.step();
        }
        CHECK(net.getSynapseWeight(0, 1) == 0.3f && net.getEligibility(0, 1) > 0,
              "pairing without reward only marks the synapse (eligibility %.4f)", net.getEligibility(0, 1));
    }
    float plus = paired(5, 50, 1.0f), minus = paired(5, 50, -1.0f), anti = paired(-5, 50, 1.0f);
    float late = paired(5, 2000, 1.0f);
    CHECK(plus > 0, "reward 50 ms after pre->post strengthens (%+.4f)", plus);
    CHECK(minus < 0, "punishment weakens it (%+.4f)", minus);
    CHECK(anti < 0, "reward after post->pre weakens (%+.4f)", anti);
    CHECK(std::fabs(late) < 0.01f * plus, "reward 2 s later has almost no effect (%+.6f)", late);

    UnifiedNetwork net(3);
    net.addNeuron(0, NeuronType::REGULAR_SPIKING);
    net.addNeuron(1, NeuronType::REGULAR_SPIKING);
    net.addNeuron(2, NeuronType::DOPAMINERGIC);
    net.enableHomeostasis(false);
    net.enableRewardModulation(true);
    net.connect(0, 1, 0.3f);
    for (int t = 0; t < 80; ++t) {
        if (t == 10) net.forceSpike(0);
        if (t == 15) net.forceSpike(1);
        if (t == 60) net.forceSpike(2);
        net.step();
    }
    CHECK(net.getSynapseWeight(0, 1) > 0.3f, "a dopaminergic neuron's spike delivers the reward (%.4f)",
          net.getSynapseWeight(0, 1));
}

static void test_homeostasis() {
    section("homeostasis");
    UnifiedNetwork net(2);
    net.addNeuron(0, NeuronType::REGULAR_SPIKING);
    net.addNeuron(1, NeuronType::REGULAR_SPIKING);
    net.enableLearning(false);
    net.connect(0, 1, 1.0f);
    float early = 0, late = 0;
    for (int t = 0; t < 60000; ++t) {
        net.stimulate(0, 20);
        net.step();
        if (t == 2000) early = net.getFiringRateHz(1);
    }
    late = net.getFiringRateHz(1);
    float target = net.config().target_rate * 1000;
    CHECK(early > 2 * target && std::fabs(late - target) < 0.5f * target,
          "an over-driven neuron scales its input down toward %.0f Hz (%.1f -> %.1f Hz, scale %.2f)", target,
          early, late, net.getHomeostaticScale(1));

    UnifiedNetwork off(2);
    off.addNeuron(0, NeuronType::REGULAR_SPIKING);
    off.addNeuron(1, NeuronType::REGULAR_SPIKING);
    off.enableLearning(false);
    off.enableHomeostasis(false);
    off.connect(0, 1, 1.0f);
    for (int t = 0; t < 60000; ++t) { off.stimulate(0, 20); off.step(); }
    CHECK(off.getFiringRateHz(1) > 2 * target, "with homeostasis off it keeps firing fast (%.1f Hz)",
          off.getFiringRateHz(1));
}

static void test_dendrites() {
    section("dendrites (phase 2)");
    auto run = [](float soma, float apical, float basal, int* plateau) {
        UnifiedNetwork net(1);
        net.addNeuron(0, NeuronType::REGULAR_SPIKING);
        net.enablePhase2(true);
        net.enableHomeostasis(false);
        int spikes = 0, pl = 0;
        for (int t = 0; t < 300; ++t) {
            if (t >= 50 && t < 250) {
                net.stimulate(0, soma);
                net.stimulateApical(0, apical);
                net.stimulateBasal(0, basal);
            }
            net.step();
            spikes += net.getNeuronSpiked(0);
            pl += net.getNeuronInPlateau(0);
        }
        if (plateau) *plateau = pl;
        return spikes;
    };
    int pa = 0, ps = 0, pb = 0;
    int apical_only = run(0, 3, 0, &pa), soma_only = run(5, 0, 0, &ps), both = run(5, 3, 0, &pb);
    CHECK(apical_only == 0 && pa == 0, "weak apical input alone: no spikes, no plateau");
    CHECK(soma_only > 0 && ps == 0, "somatic input alone: %d spikes, no plateau", soma_only);
    CHECK(pb > 0 && both > soma_only, "both together: calcium plateau, %d spikes (BAC firing)", both);
    CHECK(run(0, 0, 6, nullptr) > 0, "basal input drives the soma");

    // NMDA-like supralinearity, measured in the dendrite alone.
    auto peak = [](float basal) {
        UnifiedNetwork net(1);
        net.addNeuron(0, NeuronType::REGULAR_SPIKING);
        net.enablePhase2(true);
        net.config().g_basal = 0.0f;  // keep the soma out of it
        float rest = net.getNeuronBasalVoltage(0), top = rest;
        for (int t = 0; t < 200; ++t) {
            net.stimulateBasal(0, basal);
            net.step();
            top = std::fmax(top, net.getNeuronBasalVoltage(0));
        }
        return top - rest;
    };
    float one = peak(3), two = peak(6);
    CHECK(two > 2.2f * one, "basal inputs add supralinearly (%.1f mV, doubled input %.1f mV)", one, two);

    UnifiedNetwork net(2);
    net.addNeuron(0, NeuronType::REGULAR_SPIKING);
    net.addNeuron(1, NeuronType::REGULAR_SPIKING);
    net.enableHomeostasis(false);
    net.connect(0, 1, 1.0f);  // created in phase 1
    net.enablePhase2(true);
    float rest = net.getNeuronBasalVoltage(1);
    net.forceSpike(0);
    float top = rest;
    for (int t = 0; t < 30; ++t) { net.step(); top = std::fmax(top, net.getNeuronBasalVoltage(1)); }
    CHECK(top > rest + 5, "toggles apply to existing synapses: input now lands on the basal dendrite (+%.1f mV)",
          top - rest);
}

static void test_structural() {
    section("structural plasticity");
    UnifiedNetwork net(3);
    for (int i = 0; i < 3; ++i) net.addNeuron(i, NeuronType::REGULAR_SPIKING);
    net.enableHomeostasis(false);
    net.enableStructuralPlasticity(true);
    net.connect(2, 0, 0.01f);
    net.connect(2, 1, 0.5f);
    for (int t = 0; t < 1000; ++t) {
        if (t % 100 == 10) net.forceSpike(0);
        if (t % 100 == 15) net.forceSpike(1);
        net.step();
    }
    CHECK(!net.hasSynapse(2, 0), "a synapse below the prune floor is removed");
    CHECK(net.hasSynapse(2, 1) && net.getSynapseWeight(2, 1) == 0.5f, "a strong synapse is kept");
    CHECK(net.hasSynapse(0, 1), "0 repeatedly firing just before 1 grows a synapse 0->1");
    CHECK(!net.hasSynapse(1, 0), "... but not 1->0");
}

static void test_reset() {
    section("reset");
    UnifiedNetwork net(2);
    net.addNeuron(0, NeuronType::REGULAR_SPIKING);
    net.addNeuron(1, NeuronType::REGULAR_SPIKING);
    net.connect(0, 1, 0.1f);
    for (int t = 0; t < 30; ++t) {
        if (t == 10) net.forceSpike(0);
        if (t == 15) net.forceSpike(1);
        net.step();
    }
    float w = net.getSynapseWeight(0, 1);
    net.stimulate(0, 10);
    net.step();
    net.reset();
    CHECK(net.getSynapseWeight(0, 1) == w && w > 0.1f, "reset keeps learned weights (%.4f)", w);
    CHECK(net.getTime() == 0 && net.getSynapticCurrent(1) == 0.0f && net.getNeuronVoltage(0) < -69.0f,
          "reset clears time, currents and voltages");
}

static void test_experiment_helpers() {
    section("fixed synapses, normalisation, adaptive threshold");
    {
        UnifiedNetwork net(3);
        for (int i = 0; i < 3; ++i) net.addNeuron(i, NeuronType::REGULAR_SPIKING);
        net.enableHomeostasis(false);
        net.connect(0, 2, 0.1f);
        net.connect(1, 2, 0.1f);
        net.setPlastic(1, 2, false);
        for (int t = 0; t < 30; ++t) {
            if (t == 10) { net.forceSpike(0); net.forceSpike(1); }
            if (t == 15) net.forceSpike(2);
            net.step();
        }
        CHECK(net.getSynapseWeight(0, 2) > 0.1f && net.getSynapseWeight(1, 2) == 0.1f,
              "a fixed synapse is left alone by STDP (plastic %.4f, fixed %.4f)", net.getSynapseWeight(0, 2),
              net.getSynapseWeight(1, 2));
        float before = net.normalizeIncoming(2, 0.5f);
        float after = net.getSynapseWeight(0, 2);
        CHECK(std::fabs(before - 0.135f) < 0.01f && std::fabs(after - 0.5f) < 1e-4f && net.getSynapseWeight(1, 2) == 0.1f,
              "normalisation rescales only plastic inputs (sum %.3f -> 0.5, fixed stays 0.1)", before);
    }
    {
        auto count = [](bool adaptive, int* after_reset) {
            UnifiedNetwork net(1);
            net.addNeuron(0, NeuronType::REGULAR_SPIKING);
            net.enableHomeostasis(false);
            net.enableAdaptiveThreshold(adaptive);
            net.config().theta_plus = 0.5f;
            int first = 0, last = 0;
            for (int t = 0; t < 2000; ++t) {
                net.stimulate(0, 10);
                net.step();
                if (t < 500) first += net.getNeuronSpiked(0);
                if (t >= 1500) last += net.getNeuronSpiked(0);
            }
            net.reset();
            *after_reset = 0;
            for (int t = 0; t < 500; ++t) { net.stimulate(0, 10); net.step(); *after_reset += net.getNeuronSpiked(0); }
            return std::make_pair(first, last);
        };
        int ra = 0, rn = 0;
        auto a = count(true, &ra), n = count(false, &rn);
        CHECK(a.second < a.first / 2 && n.second * 10 >= n.first * 7,
              "adaptive threshold: firing falls %d -> %d per 500 ms (without: %d -> %d)", a.first, a.second, n.first,
              n.second);
        CHECK(ra < rn / 2, "the threshold is learned state: reset() keeps it (%d vs %d spikes)", ra, rn);
    }
}

static void test_random_network() {
    section("random E/I network");
    for (int p2 = 0; p2 < 2; ++p2) {
        const int n = 200, n_exc = 160;
        UnifiedNetwork net(n);
        net.enablePhase2(p2);
        for (int i = 0; i < n; ++i)
            net.addNeuron(i, i < n_exc ? NeuronType::REGULAR_SPIKING : NeuronType::FAST_SPIKING);
        uint32_t seed = 12345;
        auto rnd = [&seed]() { seed = seed * 1664525u + 1013904223u; return (seed >> 8) / 16777216.0f; };
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                if (i != j && rnd() < 0.1f) net.connect(i, j, i < n_exc ? 0.1f : 0.3f);
        bool finite = true;
        long spikes = 0;
        for (int t = 0; t < 5000; ++t) {
            for (int i = 0; i < n; ++i) net.stimulate(i, rnd() * 9.0f);  // noisy background input
            net.step();
            for (int i = 0; i < n; ++i) {
                spikes += net.getNeuronSpiked(i);
                finite = finite && std::isfinite(net.getNeuronVoltage(i));
            }
        }
        double hz = spikes / (double)n / 5.0;
        CHECK(finite && hz > 0.5 && hz < 100, "phase %d: 200 neurons, 5 s: finite, mean rate %.1f Hz, %d synapses",
              p2 + 1, hz, net.getNumConnections());
    }
}

int main() {
    test_neurons();
    test_synapses();
    test_stdp();
    test_reward();
    test_homeostasis();
    test_dendrites();
    test_structural();
    test_reset();
    test_experiment_helpers();
    test_random_network();
    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
