import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, dataloader, device="cuda", multiclass=False):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            preds, _ = model(xb)
            all_preds.extend(preds.argmax(1).cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    # Update class names for multi-class
    if multiclass:
        class_names = ['Normal', 'Fuzzers', 'Analysis', 'Backdoor', 'DoS',
                       'Exploits', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms']
    else:
        class_names = ["Normal", "Attack"]

    report = classification_report(all_labels, all_preds, target_names=class_names)

    print("\nValidation Accuracy:", acc)
    print("\nClassification Report:\n", report)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8) if multiclass else (5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()