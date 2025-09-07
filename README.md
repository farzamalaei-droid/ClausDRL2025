# ClausDRL2025
Repository for the DRL-ANN optimization of the Claus process

## Description
This repository contains reproducible code, example scripts, and Appendix E (hyperparameters & training details) for the Deep Reinforcement Learning-enhanced Artificial Neural Network (DRL-ANN) described in the accompanying manuscript.

## Contents
- `code/dqn_train.py`: Example DQN implementation (TensorFlow 2.x) with the specified architecture (64,32,16). This script is illustrative and uses a placeholder surrogate process model. Replace `surrogate_model.predict()` with your trained ANN or physical model for real experiments.
- `code/surrogate_model.py`: Small placeholder surrogate (synthetic) to allow running the example end-to-end.
- `AppendixE.md`: Full hyperparameters, training settings, and reproducibility checklist for the manuscript.
- `requirements.txt`: Python packages used / recommended.
- `LICENSE`: MIT license.
- `.gitignore`

## Data Availability Statement (for manuscript)
Full details, including hyperparameters, training procedures, and source code, are available in Supplementary Material (Appendix E) and the GitHub repository:  
👉 [https://github.com/farzamalaei-droid/ClausDRL2025](https://github.com/farzamalaei-droid/ClausDRL2025)

## Citation
If you use this repository, please cite the associated manuscript.
