# ClausDRL2025
**Repository for the DRL-ANN optimization of the Claus process**

**Description**
This repository contains reproducible code, example scripts, and Appendix E (hyperparameters & training details)
for the Deep Reinforcement Learning-enhanced Artificial Neural Network (DRL-ANN) described in the accompanying manuscript.

**Contents**
- `code/dqn_train.py`: Example DQN implementation (TensorFlow 2.x) with the specified architecture (64,32,16).
  This script is *illustrative* and uses a placeholder surrogate process model. Replace `surrogate_model.predict()`
  with your trained ANN or physical model for real experiments.
- `code/surrogate_model.py`: small placeholder surrogate (synthetic) to allow running the example end-to-end.
- `AppendixE.md`: full hyperparameters, training settings, and reproducibility checklist for the manuscript.
- `requirements.txt`: Python packages used / recommended.
- `LICENSE`: MIT license.
- `.gitignore`

**How to make this a real GitHub repo (recommended steps)**
1. Create a GitHub repository on your account named `ClausDRL2025` (https://github.com/<YourGitHubUsername>/ClausDRL2025)
2. From the folder where you unzipped this project do:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: DRL-ANN Claus optimization (supplementary code)"
   # create remote on GitHub via web UI or using GitHub CLI:
   gh repo create <YourGitHubUsername>/ClausDRL2025 --public --source=. --remote=origin --push
   # or (if you created repo via web UI) set remote and push:
   git remote add origin https://github.com/<YourGitHubUsername>/ClausDRL2025.git
   git branch -M main
   git push -u origin main
   ```
3. Create a release (v1.0) and optionally archive on Zenodo for a DOI (recommended for citation).

**Manuscript sentence to replace the placeholder URL**
> Full details, including hyperparameters, training procedures, and source code, are available in Supplementary Material (Appendix E) and the GitHub repository (https://github.com/<YourGitHubUsername>/ClausDRL2025).

**Note (important)**
I cannot create a GitHub repository on your behalf. The link `https://github.com/ClausDRL2025` (no username) is invalid as a repository URL.
A valid repo URL must include a user/org, e.g. `https://github.com/YourGitHubUsername/ClausDRL2025`.
Replace `<YourGitHubUsername>` with your actual GitHub username once you create the repo and push the files.
