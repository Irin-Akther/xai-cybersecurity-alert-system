# Insider Threat Detection Using Transformers

## Overview
This project implements an insider threat detection system using Transformer-based models like BERT. It combines behavioral patterns, threat intelligence data, and simulated communication logs to identify potential threats in an organizational setting.

## Features
- Fine-tuned BERT model for binary classification (Threat vs. Benign).
- Data preprocessing includes synthetic log generation and tokenization.
- Model explainability using token importance and attention heatmaps.
- Integrates real-world threat intelligence like MITRE ATT&CK and AlienVault IOCs.

## Key Components
1. **Data Collection**: 
   - Fetching techniques from MITRE ATT&CK.
   - Combining synthetic logs with real-world threat indicators.
2. **Model Training**:
   - Fine-tuned BERT model for classifying insider behavior.
   - Balancing the dataset using SMOTE.
3. **Explainability**:
   - Visualizations to understand model decisions (e.g., heatmaps).

## Code Overview
- **`final_code_for_insider_threat.py`**:
  - Preprocessing data.
  - Fine-tuning and training BERT.
  - Generating model metrics and visualizations.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repository.git
