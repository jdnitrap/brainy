#ifndef NEURON_UNIFIED_HPP
#define NEURON_UNIFIED_HPP

#include <cmath>
#include <algorithm>

enum class NeuronType {
    REGULAR_SPIKING,
    FAST_SPIKING,
    BURSTING,
    DOPAMINERGIC
};

// ============================================================================
// Unified Neuron: Phase 1 (Izhikevich) + Phase 2 (Compartmental)
// Toggle phase2_enabled to switch behavior
// ============================================================================

class UnifiedNeuron {
public:
    // Phase 1: Core Izhikevich state
    float v;              // Soma membrane potential
    float u;              // Recovery variable
    
    // Phase 2: Optional dendritic compartments
    float apical_v = -70.0f;
    float basal_v = -70.0f;
    
    // Phase 1: Izhikevich parameters
    float a, b, c, d;
    float threshold = 30.0f;
    float resting = -70.0f;
    
    // Phase 2: Dendritic coupling
    float apical_to_soma = 0.5f;
    float basal_to_soma = 0.5f;
    float backprop_strength = 0.3f;
    
    // State tracking
    bool spiked = false;
    int last_spike_time = -1000;
    int refractory_period = 0;
    
    // Feature toggles
    bool phase2_enabled = false;      // Enable dendritic compartments
    bool use_refractory = true;       // Enable refractory period
    bool use_backprop = false;        // Enable backpropagating AP (only phase 2)
    
    UnifiedNeuron(NeuronType type = NeuronType::REGULAR_SPIKING) {
        v = resting;
        u = 0.0f;
        
        switch (type) {
        case NeuronType::REGULAR_SPIKING:
            a = 0.02f; b = 0.2f; c = -65.0f; d = 8.0f;
            refractory_period = 2;
            break;
        case NeuronType::FAST_SPIKING:
            a = 0.1f; b = 0.2f; c = -65.0f; d = 2.0f;
            refractory_period = 3;
            break;
        case NeuronType::BURSTING:
            a = 0.02f; b = 0.25f; c = -55.0f; d = 0.05f;
            refractory_period = 1;
            break;
        case NeuronType::DOPAMINERGIC:
            a = 0.01f; b = 0.15f; c = -65.0f; d = 10.0f;
            refractory_period = 4;
            break;
        }
    }
    
    void integrate(float input_current, int current_time) {
        spiked = false;
        
        // Check refractory period
        if (use_refractory && refractory_period > 0 && 
            (current_time - last_spike_time) <= refractory_period) {
            return;
        }
        
        float total_input = input_current;
        
        // Phase 2: Add dendritic contributions
        if (phase2_enabled) {
            total_input += apical_to_soma * apical_v + basal_to_soma * basal_v;
        }
        
        // Izhikevich integration (two half-steps for stability)
        float step_size = 0.5f;
        
        float dv = (0.04f * v * v + 5.0f * v + 140.0f - u + total_input);
        float du = a * (b * v - u);
        
        v += step_size * dv;
        u += step_size * du;
        
        dv = (0.04f * v * v + 5.0f * v + 140.0f - u + total_input);
        du = a * (b * v - u);
        
        v += step_size * dv;
        u += step_size * du;
        
        // Check for spike
        if (v >= threshold) {
            v = c;
            u = u + d;
            spiked = true;
            last_spike_time = current_time;
            
            // Phase 2: Backpropagating action potential
            if (phase2_enabled && use_backprop) {
                apical_v += backprop_strength * (30.0f - apical_v);
                basal_v += backprop_strength * (30.0f - basal_v);
            }
        }
    }
    
    // Phase 2: Dendritic input integration
    void integrateApical(float apical_input) {
        if (phase2_enabled) {
            apical_v = 0.95f * apical_v + 0.05f * apical_input;
            apical_v = std::max(-85.0f, std::min(30.0f, apical_v));
        }
    }
    
    void integrateBasal(float basal_input) {
        if (phase2_enabled) {
            basal_v = 0.95f * basal_v + 0.05f * basal_input;
            basal_v = std::max(-85.0f, std::min(30.0f, basal_v));
        }
    }
    
    void reset_state() {
        v = resting;
        u = 0.0f;
        apical_v = resting;
        basal_v = resting;
        spiked = false;
        last_spike_time = -1000;
    }
    
    float getVoltage() const { return v; }
    float getApicalVoltage() const { return apical_v; }
    float getBasalVoltage() const { return basal_v; }
    bool hasSpiked() const { return spiked; }
    int getLastSpikeTime() const { return last_spike_time; }
    
    void enablePhase2(bool enable) {
        phase2_enabled = enable;
        if (enable) {
            use_backprop = true;
        }
    }
};

#endif // NEURON_UNIFIED_HPP
