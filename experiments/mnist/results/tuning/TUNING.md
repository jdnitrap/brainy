# Tuning log

All tuning runs: 100 excitatory neurons, labelled on 1,000 training images,
scored on **test images 0-999**. The final report therefore also gives
accuracy on test images 1,000-9,999, which no tuning decision saw.
"Spikes" and "neurons" are per training image, averaged over the last 500
training images.

## 0. First pilot: nothing learned (before any fix)

Input weights summing to 10-78, synaptic gain 30, threshold step 0.05.
Every neuron fired ~10 times for every image (~1,000 spikes/image), the
adaptive threshold climbed past 200, and at test time 93% of images got no
answer. Accuracy 8.5% (chance).

## 1. Stronger inhibition relative to input (gain 100, input total 1-3)

Spike counts first *rose* to 5,000-9,000 per image. That exposed a library
bug: strong inhibitory current pulled the membrane below about -110 mV,
where the Izhikevich quadratic made the Euler step overshoot to a spike.
Fixed in the library (membrane floor at -100 mV, test added). Rerun:

| run | input total | inhib. weight | spikes | neurons | accuracy |
|---|---|---|---|---|---|
| w1_i1 | 1 | 1.0 | 469 | 37 | 24.0% |
| w2_i1 | 2 | 1.0 | 850 | 61 | 19.0% |
| w3_i1 | 3 | 1.0 | 1106 | 82 | 20.7% |
| w2_i0.5 | 2 | 0.5 | 991 | 55 | 16.3% |

Every neuron learned the same image (`sweep1_all_neurons_same.png`: 99
copies of one "6"): too many neurons fire per image, and the learning rate
(0.01) was huge next to the average weight (~0.003).

## 2. Smaller learning rates (input total 2, 3,000 images)

| run | STDP a+ | threshold step | spikes | neurons | accuracy |
|---|---|---|---|---|---|
| A | 1e-4 | 5e-4 | 734 | 100 | 8.9% |
| B | 3e-4 | 5e-4 | 739 | 100 | 8.9% |
| C | 1e-3 | 5e-4 | 791 | 99 | 8.9% |
| D | 3e-4 | 2e-3 | 280 | 64 | 26.9% |

Still no competition. A circuit-level probe (100 neurons, near-equal steady
input) showed why: all neurons reach threshold within the 2 ms it takes
inhibition to arrive (1 ms per synapse, two synapses), and 5 ms inhibition
lets them all recover together. Added a separate, slower inhibitory time
constant to the library (`tau_syn_inh`).

## 3. Slow inhibition (20 ms), input near threshold (5,000 images)

| run | input total | a+ | threshold step | spikes | neurons | accuracy |
|---|---|---|---|---|---|---|
| E | 1 | 3e-4 | 2e-3 | 24 | 13 | 30.3% |
| F | 1.5 | 3e-4 | 2e-3 | 151 | 73 | 22.2% |
| G | 1 | 1e-3 | 2e-3 | 39 | 19 | 24.7% |
| H | 1 | 3e-4 | 5e-3 | 25 | 13 | 43.9% |

Firing became sparse and the first real receptive fields appeared
(`sweep3_H_first_receptive_fields.png`), but many neurons learned "0"
(digits with lots of ink drive neurons hardest).

## 4. Larger threshold step; per-image brightness normalisation

| run | threshold step | normalise input | spikes | neurons | accuracy |
|---|---|---|---|---|---|
| I | 1e-2 | no | 24 | 10.5 | 62.1% |
| J | 2e-2 | no | 16 | 2.8 | 69.8% |
| K | 5e-3 | yes | 26 | 12.7 | 52.6% |
| L | 1e-2 | yes | 32 | 16.1 | 61.5% |

A larger threshold step is what makes it winner-take-all (2.8 neurons per
image) and spreads neurons evenly over the digits. Normalising brightness
over-corrected (too many "1" and "7" neurons) and was dropped.

## 5. Around the best setting

| run | threshold step | a+ | input total | spikes | neurons | accuracy |
|---|---|---|---|---|---|---|
| M | 3e-2 | 3e-4 | 1 | 25 | 1.9 | 69.2% |
| N | 5e-2 | 3e-4 | 1 | 47 | 1.2 | 62.6% (12.5% no answer) |
| O | 2e-2 | 1e-3 | 1 | 58 | 1.7 | 63.9% |
| **P** | **2e-2** | **3e-4** | **1.5** | 17 | 4.0 | **71.2%** |

P was chosen (`sweep5_P_chosen.png`). J, M and P are within the
±2.8-point 95% interval of a 1,000-image test, so the choice among them is
not significant.
