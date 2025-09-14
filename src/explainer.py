import json
from utils import preprocess_unsw
from RAGExplainer import AnomalyRAGExplainer


# Test the updated implementation
if __name__ == "__main__":
    # Initialize explainer with local models
    explainer = AnomalyRAGExplainer(knowledge_base_path='knowledge.json')

    # Test with sample features
    # example_features = np.random.randn(20)  # Simulated feature vector

    print("Loading and preprocessing data...")
    X_test, _ = preprocess_unsw('../Data/UNSW-NB15/UNSW_NB15_testing-set.csv', seq_len=5)
    #

    with open('predictions.json', 'r') as file:
        data = json.load(file)

    y_test = data[0]["predicted_label"]
    conf = data[0]["confidence"]
    print("Testing RAG explainer with local Ollama models...")
    result = explainer.explain_anomaly(X_test[0][0], prediction=y_test, confidence=conf)

    print("\n" + "=" * 60)
    print("ANOMALY EXPLANATION RESULTS:")
    print("=" * 60)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"\nExplanation:\n{result['explanation']}")

    print(f"\nMost similar anomalies:")
    for anomaly in result['similar_anomalies']:
        print(f"- {anomaly['anomaly_type']} (similarity: {anomaly.get('similarity_score', 0):.3f})")
