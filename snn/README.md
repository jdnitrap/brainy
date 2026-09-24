# Unified SNN

A small spiking neural network in header-only C++17: Izhikevich neurons,
synapses with STDP, and an optional "phase 2" with dendrites, short-term
plasticity and structural plasticity. 1 step = 1 ms.

```
make            # builds ./demo and ./tests
make test       # 57 pass/fail checks
./demo          # walk-through; every number printed is measured
```

This is a repaired version of an earlier draft whose demo printed ✓ marks
over zero spikes. What was wrong and how it was fixed is at the end.

## Use

```cpp
#include "network_unified.hpp"

UnifiedNetwork net(3);
for (int i = 0; i < 3; ++i) net.addNeuron(i, NeuronType::REGULAR_SPIKING);
net.connect(0, 1, 0.1f);          // weight 0..1; 0.3 or more fires the target with one spike
net.connect(1, 2, 0.1f);

net.forceSpike(0);                // "A fires now"
for (int t = 0; t < 100; ++t) {
    net.stimulate(1, 10.0f);      // input for this step only; call again every step for steady input
    net.step();
    if (net.getNeuronSpiked(2)) { /* ... */ }
}
float w = net.getSynapseWeight(0, 1);
```

| Call | What it does |
|---|---|
| `addNeuron(id, type)` | `REGULAR_SPIKING`, `FAST_SPIKING` (inhibitory), `BURSTING`, `DOPAMINERGIC` (its spikes release reward) |
| `connect(pre, post, w [, Target])` | excitatory or inhibitory follows the presynaptic type; `Target::APICAL/BASAL/SOMA` chooses the compartment in phase 2 |
| `stimulate`, `stimulateApical`, `stimulateBasal` | input current for the next step (adds up) |
| `forceSpike(id)` | makes the neuron spike on the next step |
| `reward(d)` | dopamine: turns eligibility traces into weight changes (reward modulation on) |
| `step()`, `run(n)`, `reset()` | `reset` clears activity and keeps the learned weights |
| `getNeuronVoltage/Spiked`, `getSpikeCount`, `getFiringRateHz`, `getSynapseWeight`, `getSynapticCurrent`, `getEligibility`, ... | inspection |
| `config()` | every constant and switch (`SnnConfig` in `neuron_unified.hpp`) |

Toggles apply to the whole network at once, including synapses and neurons
created earlier or later:

| Toggle | Default | Effect |
|---|---|---|
| `enablePhase2` | off | dendrites + short-term + structural plasticity |
| `enablePSC` | on | synaptic current decays over ~5 ms; off = the same charge in one step |
| `enableRefractory` | on | 1-3 ms after each spike, depending on type |
| `enableHomeostasis` | on | scales each neuron's excitatory input toward 10 Hz over seconds |
| `enableLearning` | on | STDP on excitatory synapses |
| `enableRewardModulation` | off | STDP only marks synapses; `reward()` decides the sign |
| `enableShortTermPlasticity` | with phase 2 | Tsodyks-Markram: E→E depressing, E→I facilitating |
| `enableStructuralPlasticity` | with phase 2 | prunes synapses below 0.02, grows links between pairs that fire in order |

## The model

- **Neurons:** Izhikevich (2003), integrated in two 0.5 ms half steps, stopping
  at the spike. Each type starts at its model's true resting potential.
- **Synapses:** the current is summed per target cell and compartment, so a
  step costs O(neurons + synapses of the neurons that spiked). There is a 1 ms
  delay. Inhibitory cells subtract current from the soma.
- **STDP:** pair-based with pre and post traces (τ = 20 ms). Each spike pair
  counts once. Soft bounds: potentiation scales with (1 − w), depression
  with w.
- **Reward:** the STDP change goes to an eligibility trace (τ = 200 ms).
  `reward(d)` adds `d × trace` to the weight.
- **Homeostasis:** synaptic scaling of excitatory input, driven by a 1 s
  running firing rate.
- **Phase 2 dendrites:**
  - Basal and apical compartments; only depolarisation above rest drives the
    soma.
  - Basal input gets an NMDA-like boost, so coincident inputs add
    supralinearly.
  - A somatic spike backpropagates. Together with apical input it starts a
    calcium plateau that turns single spikes into bursts (BAC firing).
  - The plateau has a 50 ms refractory period.
- **Structural plasticity:** each neuron keeps its 8 most frequent "fired just
  before me" partners. Every second it grows at most one new input from a
  partner that fired at least twice as often as chance. Growth is capped at
  50 inputs and 50 outputs per neuron.

## What the tests show (`make test`)

| | Measured |
|---|---|
| Resting cells, both phases | stay within 0.5 mV of rest, no spikes |
| Regular spiking at input 5 / 10 / 20 | 11 / 23 / 44 Hz; fast spiking 111 Hz at 10 |
| One spike, w = 0.2 vs 1.0 | target silent vs fires once; a 10-cell chain fires in order |
| Inhibitory cell firing at 100 Hz | target drops from 23 Hz to 0 |
| STDP, pre 5 ms before / after post | +0.035 / −0.004; no drift while idle |
| A→B→C training, 50 trials | forward links 0.1 → 0.55-0.99, links back into A → 0.002; B fires at 13 ms, before its 20 ms cue; afterwards A alone brings back B and C |
| Reward 50 ms after a pairing: +1 / −1 / 2 s late | +0.021 / −0.021 / ~0 |
| Homeostasis on an over-driven cell | 32 Hz → 10 Hz over 60 s (42 Hz with it off) |
| Phase 2: apical only / soma only / both | 0 / 3 / 7 spikes (calcium plateau and bursts) |
| 200-cell E/I network, noisy input, 5 s | finite; phase 1 settles at ~15 Hz, phase 2 ~50 Hz; 5 s simulated in ~40 ms (phase 1) / ~250 ms (phase 2) |

## Limits

- Point neurons with two passive-plus-rule dendrites. This is not a
  conductance-based or morphological model, and there are no "biological
  realism" percentages.
- One fixed 1 ms synaptic delay; no axonal delays.
- Only excitatory synapses learn. Inhibitory plasticity is not modelled.
- Structural growth is bounded by the degree caps rather than by competition
  between synapses.
- Single-threaded; floats.

## What was wrong in the draft, and the fix

| Problem | Fix |
|---|---|
| A weight-1 synapse delivered 0.3 units of current for one step; a cell needs ~4 sustained, so nothing ever propagated | current is `w × 30` with a 5 ms decay, delivered every step |
| The demo stimulated for a single step, which never fires a cell | steady input or `forceSpike` |
| STDP re-applied the last spike pair on every step, so weights drifted while idle | trace-based STDP, applied once per spike |
| Phase 2 fed the dendrites' −70 mV into the soma as −70 units of current | only depolarisation above rest flows to the soma |
| Structural plasticity eroded every idle weight to 0.0001 within a few trials | prunes only synapses that learning drove below 0.02 |
| "E/I balance": the neuron type was stored but never used | `FAST_SPIKING` synapses are inhibitory |
| Reward learning could not be reached from the network | `reward()`; dopaminergic spikes release it |
| Toggles only changed existing synapses; `stimulate` overwrote synaptic input; `reset` left synapse state | one shared config; inputs add up; `reset` clears all activity |
| Each spike scanned every synapse in a `std::map` | adjacency lists per neuron |
