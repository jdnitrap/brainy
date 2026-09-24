#ifndef NEURON_UNIFIED_HPP
#define NEURON_UNIFIED_HPP

#include <algorithm>
#include <cmath>

// ============================================================================
// Units: 1 step = 1 ms. Voltages in mV. Currents are Izhikevich input units
// (a regular-spiking cell starts firing at a steady input of about 4).
// ============================================================================

enum class NeuronType {
    REGULAR_SPIKING,  // excitatory pyramidal cell
    FAST_SPIKING,     // inhibitory interneuron (its synapses subtract current)
    BURSTING,         // excitatory chattering cell (fires in bursts)
    DOPAMINERGIC      // excitatory, slow; each spike also releases reward
};

inline bool isInhibitory(NeuronType t) { return t == NeuronType::FAST_SPIKING; }

// Every switch and constant of the model, shared by the whole network. Neurons
// and synapses read it on each step, so a toggle applies to everything at
// once, including neurons and synapses created later.
struct SnnConfig {
    // ---- toggles ----
    bool phase2 = false;          // dendritic compartments (apical, basal)
    bool psc = true;              // exponential synaptic current; off = same charge in one step
    bool refractory = true;
    bool homeostasis = true;
    bool learning = true;         // STDP on excitatory synapses
    bool short_term = false;      // Tsodyks-Markram facilitation / depression
    bool structural = false;      // pruning and growth of synapses
    bool reward_modulation = false;  // STDP goes to eligibility traces, reward() applies it
    bool adaptive_threshold = false; // each spike makes the neuron harder to fire (slowly recovers)

    // ---- synapses ----
    float syn_gain = 30.0f;       // current of a weight-1 synapse at the peak of its PSC
    float tau_syn = 5.0f;         // PSC decay time (ms)
    float w_max = 1.0f;

    // ---- STDP (pair based, all-to-all, soft bounds) ----
    float a_plus = 0.05f;         // potentiation per pairing at dt = 0, scaled by (w_max - w)
    float a_minus = 0.055f;       // depression per pairing at dt = 0, scaled by w
    float tau_plus = 20.0f;
    float tau_minus = 20.0f;

    // ---- reward modulation ----
    float tau_eligibility = 200.0f;
    float reward_rate = 1.0f;     // dw = reward_rate * dopamine * eligibility
    float dopamine_per_spike = 0.5f;  // released by each DOPAMINERGIC spike

    // ---- homeostasis (synaptic scaling of excitatory input) ----
    float target_rate = 0.01f;    // spikes per step (10 Hz)
    float tau_rate = 1000.0f;     // time window of the rate estimate (ms)
    float homeostasis_rate = 0.0001f;   // per step: scaling acts over seconds, learning over ms
    float scale_min = 0.1f, scale_max = 4.0f;

    // ---- adaptive threshold (intrinsic plasticity) ----
    // Modelled as a hyperpolarising bias current theta: +theta_plus per spike,
    // decaying with tau_theta. It is learned state: reset() keeps it.
    float theta_plus = 0.05f;
    float tau_theta = 1e7f;

    // ---- structural plasticity ----
    int structural_interval = 1000;  // steps between prune / grow passes
    float prune_below = 0.02f;       // synapses weaker than this are removed
    int growth_window = 20;          // a spike of i up to this many ms before j counts as i -> j
    int growth_count = 5;            // co-activations within one interval needed to grow i -> j
    float growth_weight = 0.1f;
    int max_out_degree = 50;         // growth stops at these degrees
    int max_in_degree = 50;

    // ---- dendrites (phase 2) ----
    float tau_basal = 10.0f, tau_apical = 20.0f;
    float dend_gain = 2.5f;          // mV of steady depolarisation per unit of dendritic input
    float g_basal = 0.5f;            // soma current per mV of basal depolarisation
    float g_apical = 0.3f;           // apical is further from the soma: weaker coupling
    float g_back = 0.01f;            // soma -> dendrite coupling (per step, per mV)
    float nmda_gain = 1.5f;          // basal input is boosted up to (1 + nmda_gain)x when depolarised
    float nmda_half = 15.0f;         // ... half boost at this depolarisation (mV above rest)
    float nmda_slope = 4.0f;
    // Backpropagating spike: a brief depolarisation of the dendrites. It opens
    // the NMDA and calcium nonlinearities but is not fed back into the soma.
    float bap_basal = 25.0f;
    float bap_apical = 6.0f;         // attenuated: alone it stays below ca_threshold
    float tau_bap = 5.0f;
    float ca_threshold = 12.0f;      // apical depolarisation (mV above rest) that, within
    int bac_window = 10;             // this many ms of a somatic spike, starts a calcium plateau
    float ca_spontaneous = 35.0f;    // apical depolarisation that starts a plateau on its own
    float ca_level = 60.0f;          // plateau holds the apical dendrite this far above rest
    int ca_duration = 25;
    int ca_refractory = 50;          // calcium channels inactivate: no new plateau for this long
};

class UnifiedNeuron {
public:
    // Izhikevich (2003) parameters and state
    float a = 0.02f, b = 0.2f, c = -65.0f, d = 8.0f;
    float v = -70.0f, u = -14.0f;
    float threshold = 30.0f;
    float resting = -70.0f;         // stable fixed point of the model for this b
    int refractory_period = 2;

    // Phase 2 compartments (mV)
    float apical_v = -70.0f;
    float basal_v = -70.0f;
    int plateau_left = 0;           // steps left in an apical calcium plateau
    int plateau_end = -1000000;     // when the last plateau ended
    float bap = 0.0f;               // backpropagating spike, 1 right after a spike, decays

    bool spiked = false;
    int last_spike_time = -1000000;
    float theta = 0.0f;             // adaptive threshold bias (kept by reset_state)

    explicit UnifiedNeuron(NeuronType type = NeuronType::REGULAR_SPIKING) : type_(type) {
        switch (type) {
        case NeuronType::REGULAR_SPIKING: a = 0.02f; b = 0.20f; c = -65.0f; d = 8.0f; refractory_period = 2; break;
        case NeuronType::FAST_SPIKING:    a = 0.10f; b = 0.20f; c = -65.0f; d = 2.0f; refractory_period = 1; break;
        case NeuronType::BURSTING:        a = 0.02f; b = 0.20f; c = -50.0f; d = 2.0f; refractory_period = 1; break;
        case NeuronType::DOPAMINERGIC:    a = 0.01f; b = 0.15f; c = -65.0f; d = 10.0f; refractory_period = 3; break;
        }
        // Rest is the lower root of 0.04 v^2 + (5 - b) v + 140 = 0.
        float p = 5.0f - b;
        resting = (-p - std::sqrt(p * p - 4.0f * 0.04f * 140.0f)) / (2.0f * 0.04f);
        reset_state();
    }

    NeuronType type() const { return type_; }

    // One 1 ms step. soma_input is the total current into the soma (external
    // plus synaptic); apical_input and basal_input only matter in phase 2.
    void integrate(float soma_input, float apical_input, float basal_input,
                   int t, const SnnConfig& cfg) {
        spiked = false;
        float input = soma_input;
        if (cfg.adaptive_threshold) {
            theta *= std::exp(-1.0f / cfg.tau_theta);
            input -= theta;
        }

        if (cfg.phase2) {
            updateDendrites(apical_input, basal_input, t, cfg);
            // Current flows in from depolarised dendrites; a resting dendrite
            // adds nothing (it does not load the soma).
            input += cfg.g_basal * (basal_v - resting) + cfg.g_apical * (apical_v - resting);
        } else {
            input += apical_input + basal_input;  // no dendrites: everything lands on the soma
        }

        if (cfg.refractory && t - last_spike_time <= refractory_period) {
            v = c;
            u += a * (b * v - u);
            return;
        }

        // Two 0.5 ms half steps; stop at the spike so v cannot run away.
        for (int k = 0; k < 2 && !spiked; ++k) {
            float dv = 0.04f * v * v + 5.0f * v + 140.0f - u + input;
            float du = a * (b * v - u);
            v += 0.5f * dv;
            u += 0.5f * du;
            if (v >= threshold) spiked = true;
        }
        if (spiked) {
            v = c;
            u += d;
            last_spike_time = t;
            if (cfg.adaptive_threshold) theta += cfg.theta_plus;
            if (cfg.phase2) bap = 1.0f;
        }
    }

    // Pins the soma to a spike this step (for experiments: "make A fire now").
    void forceSpike(int t, const SnnConfig& cfg) {
        spiked = true;
        v = c;
        u += d;
        last_spike_time = t;
        if (cfg.phase2) bap = 1.0f;
    }

    void reset_state() {
        v = resting;
        u = b * resting;
        apical_v = basal_v = resting;
        plateau_left = 0;
        plateau_end = -1000000;
        bap = 0.0f;
        spiked = false;
        last_spike_time = -1000000;
    }

    // During the spike step the reported voltage is the spike peak.
    float getVoltage() const { return spiked ? threshold : v; }
    // Dendritic voltages including the backpropagating spike.
    float getApicalVoltage() const { return std::min(0.0f, apical_v + bap_apical_ * bap); }
    float getBasalVoltage() const { return std::min(0.0f, basal_v + bap_basal_ * bap); }
    bool hasSpiked() const { return spiked; }
    int getLastSpikeTime() const { return last_spike_time; }
    bool inPlateau() const { return plateau_left > 0; }

private:
    NeuronType type_;
    float bap_basal_ = 0.0f, bap_apical_ = 0.0f;  // cfg values, kept for the getters

    // Soma voltage as the dendrites see it: spike peaks and deep
    // hyperpolarisation are not passed on (the bAP is handled separately).
    float clampedSoma() const { return std::max(resting - 15.0f, std::min(-40.0f, v)); }

    void updateDendrites(float apical_input, float basal_input, int t, const SnnConfig& cfg) {
        float vs = clampedSoma();
        bap *= std::exp(-1.0f / cfg.tau_bap);
        bap_basal_ = cfg.bap_basal;
        bap_apical_ = cfg.bap_apical;

        // Basal: NMDA-like boost, coincident inputs sum supralinearly.
        float dep_b = basal_v - resting + cfg.bap_basal * bap;
        float boost = 1.0f + cfg.nmda_gain / (1.0f + std::exp(-(dep_b - cfg.nmda_half) / cfg.nmda_slope));
        float drive_b = cfg.dend_gain * basal_input * boost;
        basal_v += (-(basal_v - resting) + drive_b + cfg.g_back * (vs - basal_v) * cfg.tau_basal) / cfg.tau_basal;

        // Apical: passive, unless a calcium plateau starts (BAC firing: apical
        // input arriving within a few ms of a somatic spike, or very strong
        // apical input on its own).
        float drive_a = cfg.dend_gain * apical_input;
        apical_v += (-(apical_v - resting) + drive_a + cfg.g_back * (vs - apical_v) * cfg.tau_apical) / cfg.tau_apical;
        float dep_a = apical_v - resting + cfg.bap_apical * bap;
        if (plateau_left == 0 && t - plateau_end > cfg.ca_refractory) {
            bool bac = dep_a > cfg.ca_threshold && t - last_spike_time <= cfg.bac_window;
            if (bac || dep_a > cfg.ca_spontaneous) plateau_left = cfg.ca_duration;
        }
        if (plateau_left > 0) {
            apical_v += 0.5f * (resting + cfg.ca_level - apical_v);
            if (--plateau_left == 0) {
                apical_v = resting;  // repolarised by potassium currents
                plateau_end = t;
            }
        }

        basal_v = std::max(-90.0f, std::min(0.0f, basal_v));
        apical_v = std::max(-90.0f, std::min(0.0f, apical_v));
    }
};

#endif  // NEURON_UNIFIED_HPP
