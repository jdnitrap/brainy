#ifndef SYNAPSE_UNIFIED_HPP
#define SYNAPSE_UNIFIED_HPP

#include "neuron_unified.hpp"

#include <algorithm>
#include <cmath>

// Where a synapse lands on the postsynaptic cell. AUTO: excitatory synapses go
// to the basal dendrite in phase 2 and to the soma in phase 1; inhibitory
// synapses always go to the soma.
enum class Target { AUTO, SOMA, BASAL, APICAL };

// Per-synapse state. The synaptic current itself is summed per postsynaptic
// cell by the network (one exponential per compartment), so a synapse only
// stores its weight and the slow variables of its plasticity rules. Those are
// updated lazily from the time of the last event, so an idle synapse costs
// nothing per step.
class UnifiedSynapse {
public:
    int pre = -1, post = -1;
    float weight = 0.1f;            // 0 .. w_max; the sign comes from the presynaptic type
    bool inhibitory = false;
    Target target = Target::AUTO;

    // Short-term plasticity (Tsodyks-Markram). Default: depressing.
    float U = 0.5f;                 // release fraction of the first spike
    float tau_facil = 50.0f;
    float tau_rec = 200.0f;
    float stp_u = 0.0f, stp_x = 1.0f;
    int stp_time = -1000000;

    // Eligibility trace (reward modulation)
    float eligibility = 0.0f;
    int elig_time = 0;

    int last_pre_time = -1000000;   // for structural plasticity
    bool alive = true;

    UnifiedSynapse() = default;
    UnifiedSynapse(int pre_, int post_, float w, bool inh, Target tgt)
        : pre(pre_), post(post_), weight(w), inhibitory(inh), target(tgt) {}

    // Called when the presynaptic cell spikes. Returns the efficacy of this
    // spike relative to a rested synapse (1 when short-term plasticity is off).
    float onPreSpike(int t, const SnnConfig& cfg) {
        last_pre_time = t;
        if (!cfg.short_term) return 1.0f;
        float dt = (float)(t - stp_time);
        stp_u *= std::exp(-dt / tau_facil);
        stp_x = 1.0f - (1.0f - stp_x) * std::exp(-dt / tau_rec);
        stp_u += U * (1.0f - stp_u);
        float release = stp_u * stp_x;
        stp_x -= release;
        stp_time = t;
        return release / U;
    }

    // Weight change from STDP. With reward modulation it is held in the
    // eligibility trace until reward() arrives.
    void applyStdp(float dw, int t, const SnnConfig& cfg) {
        if (cfg.reward_modulation) {
            eligibility = eligibilityAt(t, cfg) + dw;
            elig_time = t;
        } else {
            setWeight(weight + dw, cfg);
        }
    }

    void applyReward(float dopamine, int t, const SnnConfig& cfg) {
        float e = eligibilityAt(t, cfg);
        if (e != 0.0f) setWeight(weight + cfg.reward_rate * dopamine * e, cfg);
    }

    float eligibilityAt(int t, const SnnConfig& cfg) const {
        return eligibility * std::exp(-(float)(t - elig_time) / cfg.tau_eligibility);
    }

    void resetDynamics() {
        stp_u = 0.0f;
        stp_x = 1.0f;
        stp_time = -1000000;
        eligibility = 0.0f;
        elig_time = 0;
    }

    float getWeight() const { return weight; }
    void setWeight(float w, const SnnConfig& cfg) { weight = std::max(0.0f, std::min(cfg.w_max, w)); }
    bool isAlive() const { return alive; }
};

#endif  // SYNAPSE_UNIFIED_HPP
