# Appendix E — DRL hyperparameters and training details (suggested / reproducibility checklist)

## DQN architecture
- Q-network (online): fully connected feed-forward network with 3 hidden layers: [64, 32, 16] neurons, ReLU activations.
- Output layer: linear activation with `n_actions` outputs.
- Target network: same architecture as online network; updated by periodic hard copy (every `target_update_freq` steps).

## Training hyperparameters (used in manuscript)
- Framework: TensorFlow 2.10 (tested)
- Random seed: 1234
- Episodes: 50
- Steps per episode: 100
- Replay buffer size: 10000
- Batch size: 64
- Optimizer: Adam, learning_rate = 1e-3
- Discount factor (gamma): 0.99
- Target update frequency (hard update): every 500 training steps
- Epsilon-greedy: eps_init = 1.0, eps_decay = 0.995, eps_min = 0.01
- Loss: Mean Squared Error (MSE) between target Q and predicted Q
- Reward: R = H2S_removal(%) - 0.1 * |ΔParameters|  (|ΔParameters| interpreted as L1 norm of parameter changes per step)

## Action / State space (as in manuscript)
- State: [inlet_temp (200–280 °C), inflow_rate (875–950 kmol/h), pressure (1.6–2.0 bar_abs), catalyst_surface_area (30–50 m²/g)]
- Actions: 9 discrete actions -> {no-op, ±10°C, ±25 kmol/h, ±0.1 bar_abs, ±5 m²/g} implemented as atomic parameter adjustments. (In code we map indices to delta vectors.)

## Reproducibility checklist
- Include the random seed in all libraries (TensorFlow, NumPy, Python's `random`), and save it with training logs.
- Upload the final trained model weights and a training log (loss / reward curves).
- Provide a small example dataset or the trained ANN surrogate used as the environment (or code & script to train it from raw CCD).
- Tag releases on GitHub (v1.0) and archive to Zenodo for a DOI for stable citation.

## How we reported R²
- For the manuscript, report R² on an unseen test set (not used during replay buffer population or training).
- When claiming "DQN achieves optimal conditions with an R² of 0.978" ensure that this R² refers to the ANN surrogate's predictive R² on withheld data, and not the DQN itself (DQN optimizes via the surrogate).
