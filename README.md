# Federated Anomaly Detection with Transformer

A PyTorch implementation of federated learning for network anomaly detection using Transformer architecture on the UNSW-NB15 dataset.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Federated Learning](https://img.shields.io/badge/Federated%20Learning-Enabled-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📖 Overview

This project implements a federated learning framework for network intrusion detection using a Transformer-based model. The system can operate in both centralized and federated learning modes, allowing for privacy-preserving collaborative training across multiple data silos without sharing raw data.

## ✨ Features

- **Transformer Architecture**: Utilizes self-attention mechanisms for effective anomaly detection
- **Federated Learning**: Implements FedAvg algorithm for privacy-preserving training
- **Multiple Clients**: Supports training across distributed data sources
- **UNSW-NB15 Dataset**: Preprocessing and integration with the benchmark network intrusion dataset
- **Flexible Configuration**: Easy switching between centralized and federated modes
- **Comprehensive Evaluation**: Includes accuracy, confusion matrix, and classification reports

## 🏗️ Architecture

### Model Architecture
```
AnomalyTransformer
├── Input Projection (Linear)] 
├── Transformer Encoder (Multi-head Attention)
├── Temporal Pooling (Mean)
└── Classifier (MLP)
```

### Federated Learning Flow
1. **Initialization**: Global model is initialized and distributed to clients
2. **Local Training**: Each client trains on its local data
3. **Aggregation**: Server performs federated averaging (FedAvg)
4. **Distribution**: Updated global model is sent back to clients
5. **Evaluation**: Periodic validation of global model performance

## 📦 Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA-enabled GPU (recommended)

### Install Dependencies

```bash
git clone https://github.com/your-username/federated-anomaly-detection.git
cd federated-anomaly-detection

pip install -r requirements.txt
```

## 📊 Dataset Setup
```Data
└── Data
    └── UNSW-NB15
        ├── UNSW_NB15_training-set.csv
        └── UNSW_NB15_testing-set.csv
└── src
    ├── EDA.ipynb          # Exploratory data analysis notebook
    ├── models.py          # Transformer model definition
    ├── utils.py           # Data preprocessing and dataset classes
    ├── train.py           # Training utilities
    ├── test.py            # Evaluation and metrics
    ├── federated.py       # Federated learning implementation
    ├── main.py           # Main entry point
├── requirements.txt   # Python dependencies
├── README.md
├── LICENSE
```

## 🚀 Usage

### Centralized Training
```
python main.py
# Choose option 1 when prompted
```
### Federated Training

```
python main.py
# Choose option 2 when prompted
```

#### Custom Configuration
```
# Number of clients
num_clients = 5

# Training parameters
num_rounds = 50
epochs_per_client = 3
learning_rate = 1e-4
batch_size = 512
sequence_length = 5
```
#### Sample Output
```
Federated Rounds: 100%|██████████| 50/50 [03:11<00:00,  3.82s/it]
Final Evaluation:

Validation Accuracy: 0.9247747233945477

Classification Report:
               precision    recall  f1-score   support

           0       0.80      0.97      0.88      4794
           1       0.99      0.91      0.95     12740

    accuracy                           0.92     17534
   macro avg       0.89      0.94      0.91     17534
weighted avg       0.94      0.92      0.93     17534
```