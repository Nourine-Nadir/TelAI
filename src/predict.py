import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from models import AnomalyAwareTransformer  # or your chosen model
from utils import preprocess_unsw, FlowDataset
import argparse
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_model(model_path, input_dim, device="cuda"):
    """Load trained model from checkpoint"""
    model = AnomalyAwareTransformer(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def predict_on_subset(model, test_dataloader, device="cuda"):
    """Make predictions on test data"""
    all_preds = []
    all_probs = []
    all_labels = []
    all_anomaly_scores = []

    with torch.no_grad():
        for xb, yb in test_dataloader:
            xb, yb = xb.to(device), yb.to(device)
            preds, anomaly_weights = model(xb)

            # Get predictions and probabilities
            probs = torch.softmax(preds, dim=1)
            pred_classes = preds.argmax(dim=1)

            all_preds.extend(pred_classes.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
            all_anomaly_scores.extend(anomaly_weights.cpu().numpy())

    return all_preds, all_probs, all_labels, all_anomaly_scores


def predict_single_sequence(model, sequence, device="cuda"):
    """Predict on a single sequence"""
    model.eval()
    with torch.no_grad():
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
        preds, anomaly_weights = model(sequence_tensor)
        probs = torch.softmax(preds, dim=1)
        pred_class = preds.argmax(dim=1).item()
        confidence = probs[0][pred_class].item()

    return pred_class, confidence, anomaly_weights.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='Predict network anomalies using trained model')
    parser.add_argument('--model_path', type=str, default='saved_models/federated_model_seq1_best.pth', required=False, help='Path to trained model')
    parser.add_argument('--data_path', type=str,default='../Data/UNSW-NB15/UNSW_NB15_testing-set.csv', required=False, help='Path to test CSV file')
    parser.add_argument('--seq_len', type=int, default=5, help='Sequence length')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to predict')
    parser.add_argument('--output_file', type=str, default='predictions.json', help='Output file for predictions')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for prediction')
    parser.add_argument('--multiclass', action='store_true', help='Use multi-class classification')

    args = parser.parse_args()

    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_test, y_test = preprocess_unsw(args.data_path, seq_len=args.seq_len, multiclass=args.multiclass)

    # Update class names for output
    if args.multiclass:
        class_names = ['Normal', 'Fuzzers', 'Analysis', 'Backdoor', 'DoS',
                       'Exploits', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms']
    else:
        class_names = ["Normal", "Attack"]

    # Take a subset if specified
    if args.num_samples < len(X_test):
        indices = np.random.choice(len(X_test), args.num_samples, replace=False)
        X_subset = X_test[indices]
        y_subset = y_test[indices]
    else:
        X_subset = X_test
        y_subset = y_test

    # Create dataset and dataloader
    test_dataset = FlowDataset(X_subset, y_subset)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Load model
    print("Loading model...")
    input_dim = X_test.shape[2]  # Number of features
    model = load_model(args.model_path, input_dim, device)

    # Make predictions
    print("Making predictions...")
    predictions, probabilities, true_labels, anomaly_scores = predict_on_subset(model, test_dataloader, device)

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    print(f"\nPrediction Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=["Normal", "Attack"]))

    # Save predictions to file
    results = []
    for i in range(len(predictions)):
        results.append({
            'sequence_id': i,
            'true_label': int(true_labels[i]),
            'predicted_label': int(predictions[i]),
            'confidence': float(probabilities[i][predictions[i]]),
            'normal_probability': float(probabilities[i][0]),
            'attack_probability': float(probabilities[i][1]),
            'anomaly_score': float(np.mean(anomaly_scores[i])) if anomaly_scores else 0.0
        })

    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nPredictions saved to {args.output_file}")

    # Plot confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to confusion_matrix.png")

    # Show some example predictions
    print("\nSample Predictions:")
    print("=" * 80)
    print(f"{'ID':<5} {'True':<8} {'Predicted':<10} {'Confidence':<12} {'Anomaly Score':<15}")
    print("-" * 80)
    for i in range(min(200, len(results))):
        result = results[i]
        print(
            f"{i:<5} {result['true_label']:<8} {result['predicted_label']:<10} {result['confidence']:<12.4f} {result['anomaly_score']:<15.4f}")



if __name__ == "__main__":
    main()