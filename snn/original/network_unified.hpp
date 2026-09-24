#ifndef NETWORK_UNIFIED_HPP
#define NETWORK_UNIFIED_HPP

#include "neuron_unified.hpp"
#include "synapse_unified.hpp"
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>

class UnifiedNetwork {
private:
    std::vector<UnifiedNeuron> neurons;
    std::map<std::pair<int,int>, UnifiedSynapse> synapses;
    std::vector<NeuronType> neuron_types;
    
    // Homeostatic plasticity
    std::vector<float> firing_rates;
    std::vector<float> homeostatic_scale;
    float target_firing_rate = 0.1f;
    
    // Dendritic inputs (Phase 2)
    std::vector<float> apical_input;
    std::vector<float> basal_input;
    std::vector<float> direct_input;
    
    // State
    int current_time = 0;
    
    // Feature toggles (Phase 1 vs Phase 2)
    bool phase2_enabled = false;
    bool learning_enabled = true;
    bool homeostasis_enabled = true;
    
public:
    UnifiedNetwork(int num_neurons) 
        : neurons(num_neurons),
          neuron_types(num_neurons, NeuronType::REGULAR_SPIKING),
          firing_rates(num_neurons, 0.0f),
          homeostatic_scale(num_neurons, 1.0f),
          apical_input(num_neurons, 0.0f),
          basal_input(num_neurons, 0.0f),
          direct_input(num_neurons, 0.0f) {}
    
    // ========== Configuration ==========
    
    void addNeuron(int neuron_id, NeuronType type) {
        if (neuron_id >= 0 && neuron_id < (int)neurons.size()) {
            neurons[neuron_id] = UnifiedNeuron(type);
            neurons[neuron_id].phase2_enabled = phase2_enabled;
            neuron_types[neuron_id] = type;
        }
    }
    
    void connect(int pre, int post, float weight) {
        if (pre >= 0 && pre < (int)neurons.size() &&
            post >= 0 && post < (int)neurons.size()) {
            UnifiedSynapse syn(weight);
            syn.phase2_facilitation = phase2_enabled;
            syn.phase2_structural = phase2_enabled;
            synapses[{pre, post}] = syn;
        }
    }
    
    void stimulate(int neuron_id, float current) {
        if (neuron_id >= 0 && neuron_id < (int)neurons.size()) {
            direct_input[neuron_id] = current;
        }
    }
    
    void stimulateApical(int neuron_id, float current) {
        if (neuron_id >= 0 && neuron_id < (int)neurons.size()) {
            apical_input[neuron_id] += current;
        }
    }
    
    void stimulateBasal(int neuron_id, float current) {
        if (neuron_id >= 0 && neuron_id < (int)neurons.size()) {
            basal_input[neuron_id] += current;
        }
    }
    
    // ========== Enable/Disable Features ==========
    
    void enablePhase2(bool enable) {
        phase2_enabled = enable;
        
        // Update all neurons
        for (auto& neuron : neurons) {
            neuron.enablePhase2(enable);
        }
        
        // Update all synapses
        for (auto& [key, syn] : synapses) {
            syn.enablePhase2Facilitation(enable);
            syn.enablePhase2Structural(enable);
        }
    }
    
    void enableLearning(bool enable) {
        learning_enabled = enable;
    }
    
    void enableHomeostasis(bool enable) {
        homeostasis_enabled = enable;
    }
    
    void enablePSC(bool enable) {
        for (auto& [key, syn] : synapses) {
            syn.use_psc = enable;
        }
    }
    
    void enableRefractory(bool enable) {
        for (auto& neuron : neurons) {
            neuron.use_refractory = enable;
        }
    }
    
    void enableRewardModulation(bool enable) {
        for (auto& [key, syn] : synapses) {
            syn.enablePhase2Reward(enable);
        }
    }
    
    // ========== Simulation ==========
    
    void step() {
        // Step 1: Integrate dendritic compartments (Phase 2)
        if (phase2_enabled) {
            for (int i = 0; i < (int)neurons.size(); ++i) {
                neurons[i].integrateApical(apical_input[i] * homeostatic_scale[i]);
                neurons[i].integrateBasal(basal_input[i] * homeostatic_scale[i]);
            }
        }
        
        // Step 2: Integrate soma
        for (int i = 0; i < (int)neurons.size(); ++i) {
            neurons[i].integrate(direct_input[i] * homeostatic_scale[i], current_time);
            direct_input[i] = 0.0f;
        }
        
        // Reset inputs
        if (phase2_enabled) {
            std::fill(apical_input.begin(), apical_input.end(), 0.0f);
            std::fill(basal_input.begin(), basal_input.end(), 0.0f);
        }
        
        // Step 3: Deliver spikes (with PSC)
        for (int pre = 0; pre < (int)neurons.size(); ++pre) {
            if (neurons[pre].hasSpiked()) {
                for (auto& [key, syn] : synapses) {
                    if (key.first == pre) {
                        int post = key.second;
                        syn.onPreSpike(current_time);
                        
                        float psc = syn.getPSC();
                        
                        // Distribute to compartments (Phase 2)
                        if (phase2_enabled && pre % 2 == 0) {
                            apical_input[post] += psc;
                        } else if (phase2_enabled) {
                            basal_input[post] += psc;
                        } else {
                            direct_input[post] += psc;
                        }
                    }
                }
            }
        }
        
        // Step 4: Post-spike tracking for STDP
        for (int i = 0; i < (int)neurons.size(); ++i) {
            if (neurons[i].hasSpiked()) {
                for (auto& [key, syn] : synapses) {
                    if (key.second == i) {
                        syn.onPostSpike(current_time);
                    }
                }
            }
        }
        
        // Step 5: Apply STDP
        if (learning_enabled) {
            for (auto& [key, syn] : synapses) {
                if (syn.isAlive()) {
                    syn.updateSTDP(current_time);
                }
            }
        }
        
        // Step 6: Phase 2 decay
        if (phase2_enabled) {
            for (auto& [key, syn] : synapses) {
                if (syn.isAlive()) {
                    syn.decay();
                }
            }
        }
        
        // Step 7: Homeostatic plasticity
        if (homeostasis_enabled) {
            updateHomeostasis();
        }
        
        current_time++;
    }
    
    void run(int num_steps) {
        for (int i = 0; i < num_steps; ++i) {
            step();
        }
    }
    
    void reset() {
        for (auto& neuron : neurons) {
            neuron.reset_state();
        }
        std::fill(direct_input.begin(), direct_input.end(), 0.0f);
        std::fill(apical_input.begin(), apical_input.end(), 0.0f);
        std::fill(basal_input.begin(), basal_input.end(), 0.0f);
        std::fill(firing_rates.begin(), firing_rates.end(), 0.0f);
        std::fill(homeostatic_scale.begin(), homeostatic_scale.end(), 1.0f);
        current_time = 0;
    }
    
    // ========== Data Access ==========
    
    float getNeuronVoltage(int neuron_id) {
        if (neuron_id >= 0 && neuron_id < (int)neurons.size()) {
            return neurons[neuron_id].getVoltage();
        }
        return 0.0f;
    }
    
    float getNeuronApicalVoltage(int neuron_id) {
        if (neuron_id >= 0 && neuron_id < (int)neurons.size()) {
            return neurons[neuron_id].getApicalVoltage();
        }
        return 0.0f;
    }
    
    float getNeuronBasalVoltage(int neuron_id) {
        if (neuron_id >= 0 && neuron_id < (int)neurons.size()) {
            return neurons[neuron_id].getBasalVoltage();
        }
        return 0.0f;
    }
    
    bool getNeuronSpiked(int neuron_id) {
        if (neuron_id >= 0 && neuron_id < (int)neurons.size()) {
            return neurons[neuron_id].hasSpiked();
        }
        return false;
    }
    
    float getSynapseWeight(int pre, int post) {
        auto it = synapses.find({pre, post});
        if (it != synapses.end()) {
            return it->second.getWeight();
        }
        return 0.0f;
    }
    
    void setSynapseWeight(int pre, int post, float weight) {
        auto it = synapses.find({pre, post});
        if (it != synapses.end()) {
            it->second.setWeight(weight);
        }
    }
    
    int getNumNeurons() const { return neurons.size(); }
    int getNumConnections() const { return synapses.size(); }
    float getSparsity() const {
        int total = (int)neurons.size() * (int)neurons.size();
        return (float)synapses.size() / (total > 0 ? total : 1);
    }
    float getAverageFiringRate() {
        float sum = 0.0f;
        for (float rate : firing_rates) {
            sum += rate;
        }
        return sum / (float)neurons.size();
    }
    
    bool isPhase2Enabled() const { return phase2_enabled; }
    
private:
    void updateHomeostasis() {
        for (int i = 0; i < (int)neurons.size(); ++i) {
            if (neurons[i].hasSpiked()) {
                firing_rates[i] = 0.99f * firing_rates[i] + 0.01f;
            } else {
                firing_rates[i] = 0.99f * firing_rates[i];
            }
            
            if (firing_rates[i] > target_firing_rate * 1.2f) {
                homeostatic_scale[i] *= 0.98f;
            } else if (firing_rates[i] < target_firing_rate * 0.8f) {
                homeostatic_scale[i] *= 1.02f;
            }
            
            homeostatic_scale[i] = std::max(0.5f, std::min(2.0f, homeostatic_scale[i]));
        }
    }
};

#endif // NETWORK_UNIFIED_HPP
