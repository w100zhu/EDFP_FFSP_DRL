# EDFP_FFSP_DRL
# Cooperative Multi-Agent Deep Reinforcement Learning for Stochastic Disassembly Scheduling

This repository contains the source code and calibrated industrial datasets for the paper: **"Cooperative Multi-Agent Deep Reinforcement Learning for Stochastic Disassembly Flexible Flow Shop Scheduling with Structural Uncertainty."**

## 📖 Overview

End-of-Life (EOL) product disassembly is characterized by high variability and structural uncertainty (e.g., corrosion, deformation, missing components). This project implements a **Cooperative Multi-Agent Deep Reinforcement Learning (CMA-DRL)** framework to solve the Disassembly Flexible Flow Shop Scheduling Problem (DFFSP).

Key Features:
* **Hierarchical Agents:** Decoupled Job Selection and Machine Allocation agents.
* **Physics-Informed Attention:** A Multi-Head Cross-Attention network that captures the "Economic Density" of operations.
* **Structural Uncertainty Handling:** Dynamic operation skipping logic based on real-time component integrity.

---

## 🏭 Industrial Case Study: EV Battery Pack Disassembly

To bridge the gap between theoretical modeling and industrial reality, we instantiate our framework using real-world data from the disassembly line of the **Yutong 161 Commercial Vehicle Battery Pack**.

### Data Source
The dataset located in `Data/Industrial_Case/` is derived from the **Disassembly Difficulty Score Sheets** provided by the manufacturer. It quantifies the complexity of removing specific components (e.g., High-Voltage Box, BMS, Module Array).

### Data Calibration Logic
The raw industrial data has been processed to map "Difficulty Scores" to simulation parameters:

1.  **Processing Time ($T_{ij}$)**:
    The standard processing time is linearly mapped from the manufacturer's Difficulty Score ($S \in [1, 10]$):
    $$T_{ij} = T_{base} + \alpha \cdot S$$
    *Where higher scores indicate complex operations (e.g., removing rusted bolts) requiring longer processing times.*

2.  **Skipping Probability (Uncertainty)**:
    Components flagged with specific attributes in the raw data (e.g., "Non-destructive Access Impossible") are used to initialize the structural uncertainty probabilities in the environment.

### Dataset File Structure
* `Yutong161_Difficulty_Scores.csv`: The sanitized dataset containing Component IDs, Disassembly Methods, and Difficulty Ratings.
* `config_yutong.json`: Configuration file mapping the physical battery pack stages to the DFFSP mathematical model.

---

## 📂 Repository Structure

```text
.
├── agents/                 # Implementation of Job and Machine Agents (PPO)
├── environment/            # DFFSP Environment with Skipping Logic
│   ├── disassembly_env.py  # Main environment wrapper
│   └── skipping_logic.py   # Structural uncertainty handling
├── network/                # Multi-Head Cross-Attention Network (Fig. 3 in paper)
├── Data/
│   ├── Synthetic/          # Standard J*S*M benchmark instances
│   └── Industrial_Case/    # Yutong 161 Battery Pack Data (Calibrated)
├── main.py                 # Training entry point
├── utils.py                # Data loading and metrics calculation
└── README.md
