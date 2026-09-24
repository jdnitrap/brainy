#ifndef SYNAPSE_UNIFIED_HPP
#define SYNAPSE_UNIFIED_HPP

#include <cmath>
#include <algorithm>

// ============================================================================
// Unified Synapse: Phase 1 + Phase 2 features (all toggleable)
// ============================================================================

class UnifiedSynapse {
public:
    float weight = 0.0f;
    
    // PSC (Post-Synaptic Current) filtering
    float psc_current = 0.0f;
    float psc_decay = 0.9f;          // Phase 1
    float psc_rise_rate = 0.3f;
    
    // STDP parameters
    float eta = 0.01f;
    float A_plus = 0.01f;
    float A_minus = 0.01f;
    float tau_plus = 20.0f;
    float tau_minus = 20.0f;
    
    // Spike tracking
    int last_pre_spike = -1000;
    int last_post_spike = -1000;
    
    // Weight bounds
    float w_min = -1.0f;
    float w_max = 1.0f;
    
    // Phase 2: Short-term plasticity
    float use_fraction = 0.5f;
    float max_use_fraction = 1.0f;
    float recovery_rate = 0.01f;
    float facilitation_strength = 1.0f;
    float facilitation_decay = 0.98f;
    float max_facilitation = 1.5f;
    
    // Phase 2: Eligibility trace (for reward modulation)
    float eligibility_trace = 0.0f;
    float trace_decay = 0.95f;
    
    // Phase 2: Structural plasticity
    float activity_level = 0.0f;
    float activity_decay = 0.99f;
    float growth_rate = 0.01f;
    float min_weight = 0.0001f;
    bool alive = true;
    
    // Feature toggles
    bool phase2_facilitation = false;      // Enable short-term plasticity
    bool phase2_structural = false;         // Enable growth/pruning
    bool phase2_reward_modulation = false;  // Enable eligibility traces
    bool use_psc = true;                    // Phase 1: PSC filtering
    bool use_stdp = true;                   // Always available
    
    UnifiedSynapse(float init_weight = 0.1f) : weight(init_weight) {}
    
    void onPreSpike(int current_time) {
        last_pre_spike = current_time;
        
        if (!alive) return;
        
        // Phase 1: Basic PSC
        float strength = weight;
        
        // Phase 2: Facilitation modulates strength
        if (phase2_facilitation) {
            use_fraction *= 0.9f;  // Depression: use transmitter
            use_fraction = std::max(0.1f, use_fraction);
            
            facilitation_strength = std::min(max_facilitation, 
                                             facilitation_strength * 1.1f);
            
            strength = weight * facilitation_strength * use_fraction;
            eligibility_trace = 1.0f;  // Mark for reward learning
        }
        
        // PSC output
        if (use_psc) {
            psc_current += strength * psc_rise_rate;
        } else {
            psc_current += strength;  // Direct delivery if PSC off
        }
        
        // Phase 2: Activity tracking
        if (phase2_structural) {
            activity_level = 1.0f;
        }
    }
    
    float getPSC() {
        float output = psc_current;
        if (use_psc) {
            psc_current *= psc_decay;
        }
        return output;
    }
    
    void onPostSpike(int current_time) {
        last_post_spike = current_time;
    }
    
    void updateSTDP(int current_time) {
        if (!use_stdp || !alive) return;
        
        float dt = (float)(last_post_spike - last_pre_spike);
        
        if (std::abs(dt) > 100.0f) return;
        
        float dw = 0.0f;
        
        if (dt > 0) {
            dw = eta * A_plus * std::exp(-dt / tau_plus);
        } else if (dt < 0) {
            dw = -eta * A_minus * std::exp(dt / tau_minus);
        }
        
        weight += dw;
        weight = std::max(w_min, std::min(w_max, weight));
    }
    
    void decay() {
        if (!alive) return;
        
        // Phase 2: Facilitation decay
        if (phase2_facilitation) {
            psc_current *= psc_decay;
            
            // Recovery of transmitter
            use_fraction += recovery_rate * (max_use_fraction - use_fraction);
            
            // Facilitation decay
            facilitation_strength *= facilitation_decay;
            facilitation_strength = std::max(1.0f, facilitation_strength);
            
            // Eligibility trace decay
            eligibility_trace *= trace_decay;
        }
        
        // Phase 2: Structural plasticity
        if (phase2_structural) {
            // Growth and pruning
            if (activity_level > 0.5f && weight < w_max) {
                weight += growth_rate;
                weight = std::min(w_max, weight);
            } else if (activity_level < 0.1f && weight > min_weight) {
                weight -= growth_rate * 0.5f;
                weight = std::max(min_weight, weight);
            }
            
            // Decay activity
            activity_level *= activity_decay;
            
            // Mark for pruning if too weak
            if (weight < min_weight * 2.0f && activity_level < 0.01f) {
                alive = false;
            }
        }
    }
    
    void applyReward(float dopamine) {
        if (!phase2_reward_modulation || !alive) return;
        
        if (std::abs(dopamine) > 0.001f) {
            float dw = dopamine * eligibility_trace;
            weight += eta * dw;
            weight = std::max(w_min, std::min(w_max, weight));
        }
    }
    
    float getWeight() const { return weight; }
    void setWeight(float w) { 
        weight = std::max(w_min, std::min(w_max, w)); 
    }
    bool isAlive() const { return alive; }
    
    void enablePhase2Facilitation(bool enable) {
        phase2_facilitation = enable;
    }
    
    void enablePhase2Structural(bool enable) {
        phase2_structural = enable;
    }
    
    void enablePhase2Reward(bool enable) {
        phase2_reward_modulation = enable;
    }
};

#endif // SYNAPSE_UNIFIED_HPP
