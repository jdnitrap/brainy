#ifndef NETWORK_UNIFIED_HPP
#define NETWORK_UNIFIED_HPP

#include "neuron_unified.hpp"
#include "synapse_unified.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <unordered_map>
#include <utility>
#include <vector>

class UnifiedNetwork {
public:
    explicit UnifiedNetwork(int num_neurons)
        : neurons_(num_neurons),
          out_(num_neurons), in_(num_neurons),
          ext_soma_(num_neurons, 0.0f), ext_apical_(num_neurons, 0.0f), ext_basal_(num_neurons, 0.0f),
          forced_(num_neurons, 0),
          syn_exc_soma_(num_neurons, 0.0f), syn_inh_soma_(num_neurons, 0.0f),
          syn_basal_(num_neurons, 0.0f), syn_apical_(num_neurons, 0.0f),
          pre_trace_(num_neurons, 0.0f), post_trace_(num_neurons, 0.0f),
          rate_(num_neurons, 0.0f), scale_(num_neurons, 1.0f),
          spike_count_(num_neurons, 0), cand_((size_t)num_neurons * kCand), interval_spikes_(num_neurons, 0) {}

    // ========== Configuration ==========

    SnnConfig& config() { return cfg_; }
    const SnnConfig& config() const { return cfg_; }

    void addNeuron(int id, NeuronType type) {
        if (!valid(id)) return;
        neurons_[id] = UnifiedNeuron(type);
        for (int s : out_[id]) syn_[s].inhibitory = isInhibitory(type);
    }

    // weight is 0..1 (a weight-1 synapse peaks at config().syn_gain of
    // current). Excitatory or inhibitory follows the presynaptic neuron's type.
    // Connecting an existing pair replaces its weight.
    void connect(int pre, int post, float weight, Target target = Target::AUTO) {
        if (!valid(pre) || !valid(post)) return;
        auto it = index_.find(key(pre, post));
        if (it != index_.end()) {
            syn_[it->second].setWeight(weight, cfg_);
            syn_[it->second].target = target;
            return;
        }
        UnifiedSynapse s(pre, post, 0.0f, isInhibitory(neurons_[pre].type()), target);
        s.setWeight(weight, cfg_);
        if (!s.inhibitory && isInhibitory(neurons_[post].type())) {
            s.U = 0.1f; s.tau_facil = 500.0f; s.tau_rec = 100.0f;  // E -> I: facilitating
        }
        int idx = (int)syn_.size();
        syn_.push_back(s);
        out_[pre].push_back(idx);
        in_[post].push_back(idx);
        index_[key(pre, post)] = idx;
    }

    // Tsodyks-Markram parameters of one synapse (U: first release fraction).
    void setShortTermPlasticity(int pre, int post, float U, float tau_facil, float tau_rec) {
        if (UnifiedSynapse* s = find(pre, post)) {
            s->U = U; s->tau_facil = tau_facil; s->tau_rec = tau_rec;
            s->resetDynamics();
        }
    }

    // Fixed synapses (wiring such as lateral inhibition) are not changed by
    // STDP, reward or pruning.
    void setPlastic(int pre, int post, bool plastic) {
        if (UnifiedSynapse* s = find(pre, post)) s->plastic = plastic;
    }

    // Rescales the plastic excitatory inputs of `post` so their weights sum to
    // `total` (weight normalisation, as in Diehl & Cook 2015). Returns the
    // sum before scaling.
    float normalizeIncoming(int post, float total) {
        if (!valid(post)) return 0.0f;
        float sum = 0.0f;
        for (int s : in_[post]) if (syn_[s].plastic && !syn_[s].inhibitory) sum += syn_[s].weight;
        if (sum > 0.0f)
            for (int s : in_[post])
                if (syn_[s].plastic && !syn_[s].inhibitory) syn_[s].setWeight(syn_[s].weight * total / sum, cfg_);
        return sum;
    }

    void enableAdaptiveThreshold(bool on) { cfg_.adaptive_threshold = on; }
    float getThreshold(int id) const { return valid(id) ? neurons_[id].theta : 0.0f; }

    // External input for the next step only (adds up if called twice).
    void stimulate(int id, float current) { if (valid(id)) ext_soma_[id] += current; }
    void stimulateApical(int id, float current) { if (valid(id)) ext_apical_[id] += current; }
    void stimulateBasal(int id, float current) { if (valid(id)) ext_basal_[id] += current; }
    // Makes the neuron spike on the next step, whatever its input.
    void forceSpike(int id) { if (valid(id)) forced_[id] = 1; }

    // Global neuromodulator: turns eligibility traces into weight changes.
    void reward(float dopamine) { pending_reward_ += dopamine; }

    // ========== Feature toggles ==========

    // Phase 2 = dendrites + short-term plasticity + structural plasticity.
    // The individual toggles below can change any of them afterwards.
    void enablePhase2(bool on) { cfg_.phase2 = on; cfg_.short_term = on; cfg_.structural = on; }
    void enableLearning(bool on) { cfg_.learning = on; }
    void enableHomeostasis(bool on) { cfg_.homeostasis = on; }
    void enablePSC(bool on) { cfg_.psc = on; }
    void enableRefractory(bool on) { cfg_.refractory = on; }
    void enableRewardModulation(bool on) { cfg_.reward_modulation = on; }
    void enableShortTermPlasticity(bool on) { cfg_.short_term = on; }
    void enableStructuralPlasticity(bool on) { cfg_.structural = on; }

    // ========== Simulation ==========

    void step() {
        const int n = (int)neurons_.size();
        const int t = time_;

        // 1. Integrate every neuron on the synaptic current left by earlier
        //    spikes (1 ms delay) plus this step's external input.
        for (int i = 0; i < n; ++i) {
            float sc = cfg_.homeostasis ? scale_[i] : 1.0f;
            float soma = ext_soma_[i] + sc * syn_exc_soma_[i] - syn_inh_soma_[i];
            float basal = ext_basal_[i] + sc * syn_basal_[i];
            float apical = ext_apical_[i] + sc * syn_apical_[i];
            if (forced_[i]) neurons_[i].forceSpike(t, cfg_);
            else neurons_[i].integrate(soma, apical, basal, t, cfg_);
        }
        std::fill(ext_soma_.begin(), ext_soma_.end(), 0.0f);
        std::fill(ext_apical_.begin(), ext_apical_.end(), 0.0f);
        std::fill(ext_basal_.begin(), ext_basal_.end(), 0.0f);
        std::fill(forced_.begin(), forced_.end(), 0);

        // 2. Synaptic currents decay (PSC) or were a one-step pulse (no PSC).
        float keep = cfg_.psc ? std::exp(-1.0f / cfg_.tau_syn) : 0.0f;
        float charge = cfg_.psc ? 1.0f : 1.0f / (1.0f - std::exp(-1.0f / cfg_.tau_syn));
        for (int i = 0; i < n; ++i) {
            syn_exc_soma_[i] *= keep; syn_inh_soma_[i] *= keep;
            syn_basal_[i] *= keep; syn_apical_[i] *= keep;
        }
        float decay_plus = std::exp(-1.0f / cfg_.tau_plus);
        float decay_minus = std::exp(-1.0f / cfg_.tau_minus);
        for (int i = 0; i < n; ++i) { pre_trace_[i] *= decay_plus; post_trace_[i] *= decay_minus; }

        // 3. Spikes: deliver current, STDP, bookkeeping.
        float rate_keep = 1.0f - 1.0f / cfg_.tau_rate;
        for (int i = 0; i < n; ++i) {
            bool fired = neurons_[i].hasSpiked();
            rate_[i] = rate_keep * rate_[i] + (fired ? 1.0f / cfg_.tau_rate : 0.0f);
            if (!fired) continue;
            ++spike_count_[i];
            if (neurons_[i].type() == NeuronType::DOPAMINERGIC && cfg_.reward_modulation)
                pending_reward_ += cfg_.dopamine_per_spike;

            for (int s : out_[i]) {
                UnifiedSynapse& syn = syn_[s];
                if (!syn.alive) continue;
                float amount = syn.weight * cfg_.syn_gain * syn.onPreSpike(t, cfg_) * charge;
                deliver(syn, amount);
                // pre after post: depression, proportional to w
                if (cfg_.learning && syn.plastic && !syn.inhibitory && post_trace_[syn.post] > 0.0f)
                    syn.applyStdp(-cfg_.a_minus * post_trace_[syn.post] * syn.weight, t, cfg_);
            }
            if (cfg_.learning) {
                for (int s : in_[i]) {
                    UnifiedSynapse& syn = syn_[s];
                    // post after pre: potentiation, proportional to (w_max - w)
                    if (syn.alive && syn.plastic && !syn.inhibitory && pre_trace_[syn.pre] > 0.0f)
                        syn.applyStdp(cfg_.a_plus * pre_trace_[syn.pre] * (cfg_.w_max - syn.weight), t, cfg_);
                }
            }
            if (cfg_.structural) noteCoactivity(i, t);
        }
        // Traces are raised after the pairing, so a neuron never pairs with
        // itself and simultaneous spikes do not count.
        for (int i = 0; i < n; ++i) {
            if (neurons_[i].hasSpiked()) { pre_trace_[i] += 1.0f; post_trace_[i] += 1.0f; }
        }

        // 4. Reward: convert eligibility into weight change.
        if (pending_reward_ != 0.0f) {
            if (cfg_.reward_modulation)
                for (auto& syn : syn_)
                    if (syn.alive && syn.plastic && !syn.inhibitory) syn.applyReward(pending_reward_, t, cfg_);
            pending_reward_ = 0.0f;
        }

        // 5. Homeostatic synaptic scaling toward the target rate.
        if (cfg_.homeostasis) {
            for (int i = 0; i < n; ++i) {
                scale_[i] *= 1.0f + cfg_.homeostasis_rate * (cfg_.target_rate - rate_[i]) / cfg_.target_rate;
                scale_[i] = std::max(cfg_.scale_min, std::min(cfg_.scale_max, scale_[i]));
            }
        }

        // 6. Structural plasticity: prune weak synapses, grow co-active pairs.
        if (cfg_.structural && (t + 1) % cfg_.structural_interval == 0) restructure(t);

        ++time_;
    }

    void run(int num_steps) { for (int i = 0; i < num_steps; ++i) step(); }

    // Clears activity (voltages, currents, traces, short-term state, rate
    // estimates, clock). Weights and connections are what was learned, so they
    // stay.
    void reset() {
        for (auto& nr : neurons_) nr.reset_state();
        for (auto& s : syn_) s.resetDynamics();
        for (auto* v : {&ext_soma_, &ext_apical_, &ext_basal_, &syn_exc_soma_, &syn_inh_soma_,
                        &syn_basal_, &syn_apical_, &pre_trace_, &post_trace_, &rate_})
            std::fill(v->begin(), v->end(), 0.0f);
        std::fill(forced_.begin(), forced_.end(), 0);
        std::fill(scale_.begin(), scale_.end(), 1.0f);
        recent_.clear();
        std::fill(cand_.begin(), cand_.end(), Candidate());
        std::fill(interval_spikes_.begin(), interval_spikes_.end(), 0);
        pending_reward_ = 0.0f;
        time_ = 0;
    }

    // ========== Data access ==========

    float getNeuronVoltage(int id) const { return valid(id) ? neurons_[id].getVoltage() : 0.0f; }
    float getNeuronApicalVoltage(int id) const { return valid(id) ? neurons_[id].getApicalVoltage() : 0.0f; }
    float getNeuronBasalVoltage(int id) const { return valid(id) ? neurons_[id].getBasalVoltage() : 0.0f; }
    bool getNeuronSpiked(int id) const { return valid(id) && neurons_[id].hasSpiked(); }
    bool getNeuronInPlateau(int id) const { return valid(id) && neurons_[id].inPlateau(); }
    long getSpikeCount(int id) const { return valid(id) ? spike_count_[id] : 0; }
    float getHomeostaticScale(int id) const { return valid(id) ? scale_[id] : 0.0f; }
    // Recent firing rate in Hz (running average over config().tau_rate ms).
    float getFiringRateHz(int id) const { return valid(id) ? rate_[id] * 1000.0f : 0.0f; }

    bool hasSynapse(int pre, int post) const { return index_.count(key(pre, post)) != 0; }
    float getSynapseWeight(int pre, int post) const {
        auto it = index_.find(key(pre, post));
        return it == index_.end() ? 0.0f : syn_[it->second].weight;
    }
    void setSynapseWeight(int pre, int post, float w) {
        if (UnifiedSynapse* s = find(pre, post)) s->setWeight(w, cfg_);
    }
    // Synaptic current waiting for the next step (excitatory minus
    // inhibitory, soma plus dendrites in soma-equivalent units).
    float getSynapticCurrent(int id) const {
        if (!valid(id)) return 0.0f;
        float dend = (syn_basal_[id] + syn_apical_[id]) / toDendrite();
        return syn_exc_soma_[id] - syn_inh_soma_[id] + dend;
    }
    float getEligibility(int pre, int post) const {
        auto it = index_.find(key(pre, post));
        return it == index_.end() ? 0.0f : syn_[it->second].eligibilityAt(time_, cfg_);
    }

    int getNumNeurons() const { return (int)neurons_.size(); }
    int getNumConnections() const { return (int)index_.size(); }
    int getTime() const { return time_; }
    float getSparsity() const {
        double total = (double)neurons_.size() * (double)neurons_.size();
        return total > 0 ? (float)(index_.size() / total) : 0.0f;
    }
    // Mean recent firing rate as spikes per step (0.01 = 10 Hz).
    float getAverageFiringRate() const {
        if (rate_.empty()) return 0.0f;
        double sum = 0.0;
        for (float r : rate_) sum += r;
        return (float)(sum / rate_.size());
    }
    bool isPhase2Enabled() const { return cfg_.phase2; }

private:
    SnnConfig cfg_;
    std::vector<UnifiedNeuron> neurons_;
    std::vector<UnifiedSynapse> syn_;
    std::vector<std::vector<int>> out_, in_;
    std::unordered_map<uint64_t, int> index_;

    std::vector<float> ext_soma_, ext_apical_, ext_basal_;
    std::vector<char> forced_;
    std::vector<float> syn_exc_soma_, syn_inh_soma_, syn_basal_, syn_apical_;
    std::vector<float> pre_trace_, post_trace_;
    std::vector<float> rate_, scale_;
    std::vector<long> spike_count_;
    float pending_reward_ = 0.0f;
    int time_ = 0;

    // structural plasticity bookkeeping
    std::deque<std::pair<int, int>> recent_;  // (excitatory neuron, spike time), last growth_window ms
    // Per neuron, the kCand presynaptic candidates that most often fired just
    // before it (Misra-Gries heavy hitters: counts are lower bounds).
    static constexpr int kCand = 8;
    struct Candidate { int pre = -1; int count = 0; };
    std::vector<Candidate> cand_;
    std::vector<int> interval_spikes_;           // spikes per neuron in this interval

    bool valid(int id) const { return id >= 0 && id < (int)neurons_.size(); }
    static uint64_t key(int pre, int post) { return ((uint64_t)(uint32_t)pre << 32) | (uint32_t)post; }

    UnifiedSynapse* find(int pre, int post) {
        auto it = index_.find(key(pre, post));
        return it == index_.end() ? nullptr : &syn_[it->second];
    }

    void deliver(const UnifiedSynapse& syn, float amount) {
        int p = syn.post;
        if (syn.inhibitory) { syn_inh_soma_[p] += amount; return; }
        Target tgt = syn.target;
        if (tgt == Target::AUTO) tgt = cfg_.phase2 ? Target::BASAL : Target::SOMA;
        if (!cfg_.phase2) tgt = Target::SOMA;  // no dendrites in phase 1
        switch (tgt) {
        case Target::BASAL:  syn_basal_[p] += amount * toDendrite(); break;
        case Target::APICAL: syn_apical_[p] += amount * toDendrite(); break;
        default:             syn_exc_soma_[p] += amount; break;
        }
    }

    // Dendritic input is in dendritic units (dend_gain mV of steady
    // depolarisation each, which becomes g_basal current per mV at the soma).
    // Scaled so a steady current reaching the basal dendrite ends up as the
    // same current at the soma, before the NMDA boost.
    float toDendrite() const { return 1.0f / (cfg_.dend_gain * cfg_.g_basal); }

    void noteCoactivity(int j, int t) {
        while (!recent_.empty() && t - recent_.front().second > cfg_.growth_window) recent_.pop_front();
        Candidate* c = &cand_[(size_t)j * kCand];
        for (auto& [i, ti] : recent_) {
            if (i == j || ti == t) continue;
            int k = 0, free_slot = -1;
            for (; k < kCand && c[k].pre != i; ++k)
                if (free_slot < 0 && c[k].count == 0) free_slot = k;
            if (k < kCand) ++c[k].count;
            else if (free_slot >= 0) c[free_slot] = {i, 1};
            else for (k = 0; k < kCand; ++k) --c[k].count;
        }
        if (!isInhibitory(neurons_[j].type())) recent_.emplace_back(j, t);
        ++interval_spikes_[j];
    }

    void restructure(int t) {
        // Grow: for each neuron, the one unconnected partner that most often
        // fired just before it in this interval, if often enough and at least
        // twice as often as chance (i's rate x window x j's spikes).
        const int n = (int)neurons_.size();
        for (int j = 0; j < n; ++j) {
            Candidate* c = &cand_[(size_t)j * kCand];
            int best = -1;
            for (int k = 0; k < kCand; ++k) {
                if (c[k].count < cfg_.growth_count || hasSynapse(c[k].pre, j)) continue;
                float chance = rate_[c[k].pre] * cfg_.growth_window * interval_spikes_[j];
                if (c[k].count >= 2.0f * chance && (best < 0 || c[k].count > c[best].count)) best = k;
            }
            if (best >= 0) {
                int i = c[best].pre;
                if ((int)out_[i].size() < cfg_.max_out_degree && (int)in_[j].size() < cfg_.max_in_degree) {
                    connect(i, j, cfg_.growth_weight);
                    syn_[index_[key(i, j)]].last_pre_time = t;
                }
            }
            for (int k = 0; k < kCand; ++k) c[k] = Candidate();
        }
        std::fill(interval_spikes_.begin(), interval_spikes_.end(), 0);

        // Prune: excitatory synapses that learning has driven below the floor.
        bool removed = false;
        for (auto& s : syn_) {
            if (s.alive && s.plastic && !s.inhibitory && s.weight < cfg_.prune_below) {
                s.alive = false;
                removed = true;
            }
        }
        if (!removed) return;
        std::vector<UnifiedSynapse> kept;
        kept.reserve(syn_.size());
        for (auto& s : syn_) if (s.alive) kept.push_back(s);
        syn_.swap(kept);
        for (auto& v : out_) v.clear();
        for (auto& v : in_) v.clear();
        index_.clear();
        for (int s = 0; s < (int)syn_.size(); ++s) {
            out_[syn_[s].pre].push_back(s);
            in_[syn_[s].post].push_back(s);
            index_[key(syn_[s].pre, syn_[s].post)] = s;
        }
    }
};

#endif  // NETWORK_UNIFIED_HPP
