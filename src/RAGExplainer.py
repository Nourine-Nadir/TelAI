import numpy as np
import json
import ollama
from typing import List, Dict, Any
import time


class AnomalyRAGExplainer:
    def __init__(self, knowledge_base_path: str = None, profiles_path: str = None):
        # Use local Ollama models instead of HF models
        self.embedding_model_name = "nomic-embed-text"
        self.generator_model_name = "deepseek-r1:14b"

        # Load or create knowledge base
        print(f'knowledge_base_path: {knowledge_base_path}')
        self.knowledge_base = self._load_knowledge_base(knowledge_base_path)
        self.feature_interpretations = self.knowledge_base.get("feature_interpretations", {})
        self.anomaly_types = self.knowledge_base.get("knowledge_base", [])

        self.attack_profiles = {}
        if profiles_path:
            self._load_attack_profiles(profiles_path)

        self.embeddings = None

        if self.knowledge_base:
            self._embed_knowledge_base()

    def _embed_text(self, text: str) -> List[float]:
        """Embed text using local nomic-embed-text model"""
        try:
            response = ollama.embeddings(model=self.embedding_model_name, prompt=text)
            return response['embedding']
        except Exception as e:
            print(f"Embedding error: {e}")
            # Fallback to random embeddings
            return np.random.randn(384).tolist()  # nomic-embed-text has 384 dims

    def _generate_text(self, prompt: str, max_length: int = 300) -> str:
        """Generate text using local deepseek-r1:14b model"""
        try:
            response = ollama.generate(
                model=self.generator_model_name,
                prompt=prompt,
                options={
                    "temperature": 0.3,  # Lower temperature for more focused responses
                    "top_p": 0.9,
                    "max_length": max_length
                    # "stop": ["\n\n", "###", "Explanation:"]
                }
            )
            return response['response'].strip()
        except Exception as e:
            print(f"Generation error: {e}")
            return self._generate_fallback_explanation(1, 0.9, {})

    def _load_knowledge_base(self, path: str) -> List[Dict]:
        """Load anomaly knowledge base for UNSW-NB15 attack categories"""
        if path:
            try:
                with open(path, 'r') as f:
                    knowledge_base = json.load(f)
                    print('Loading knowledge base... ', knowledge_base)
                    return knowledge_base

            except:
                print("Could not load knowledge base, using UNSW-NB15 default")

    def _load_feature_interpretations(self, path: str) -> List[Dict]:
        if path:
            try:
                with open(path, 'r') as f:
                    feature_desc = json.load(f)['feature_interpretations']
                    print('Loading feature_interpretations... ', feature_desc)
                    return feature_desc

            except:
                print("Could not load feature_interpretations, using UNSW-NB15 default")

    def _embed_knowledge_base(self):
        """Create embeddings for anomaly types only"""
        print("Embedding anomaly knowledge base with nomic-embed-text...")
        self.embeddings = []

        for item in self.anomaly_types:
            text = f"{item['anomaly_type']}: {item['description']}. Features: {', '.join(item['features'])}"
            embedding = self._embed_text(text)
            self.embeddings.append(embedding)
            time.sleep(0.1)

        self.embeddings = np.array(self.embeddings)
        print("Anomaly knowledge base embedding completed!")

    def _load_attack_profiles(self, profiles_path: str):
        """Load precomputed attack profiles"""
        try:
            with open(profiles_path, 'r') as f:
                profiles_data = json.load(f)

            self.attack_profiles = profiles_data.get('feature_profiles', {})
            self.feature_means = np.array(profiles_data.get('feature_means', []))
            self.feature_stds = np.array(profiles_data.get('feature_stds', []))

            print(f"Loaded attack profiles for {len(self.attack_profiles)} types")

        except Exception as e:
            print(f"Could not load attack profiles: {e}")
            self.attack_profiles = {}

    def retrieve_similar_anomalies(self, feature_vector: np.array, top_k: int = 3) -> List[Dict]:
        """Retrieve similar anomalies using actual feature profiles"""
        if not self.attack_profiles:
            return self._fallback_retrieval(top_k)

        # Normalize the input feature vector using stored statistics
        if hasattr(self, 'feature_means') and hasattr(self, 'feature_stds'):
            if len(feature_vector) == len(self.feature_means):
                feature_vector = (feature_vector - self.feature_means) / self.feature_stds

        # Compute absolute values (abnormality scores)
        abs_features = np.abs(feature_vector)
        feature_profile = abs_features / (np.sum(abs_features) + 1e-8)

        # Calculate similarities with each attack profile
        similarities = {}
        for attack_type, profile_data in self.attack_profiles.items():
            if attack_type == 'Normal':
                continue  # Skip normal traffic

            attack_profile = np.array(profile_data['profile'])

            # Ensure profiles have same length
            min_len = min(len(feature_profile), len(attack_profile))
            feat_profile_trunc = feature_profile[:min_len]
            attack_profile_trunc = attack_profile[:min_len]

            # Calculate cosine similarity
            similarity = np.dot(feat_profile_trunc, attack_profile_trunc) / (
                    np.linalg.norm(feat_profile_trunc) * np.linalg.norm(attack_profile_trunc) + 1e-8
            )

            similarities[attack_type] = similarity

        # Get top-k most similar attacks
        sorted_attacks = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for attack_type, similarity in sorted_attacks:
            # Find matching anomaly in knowledge base
            for anomaly in self.anomaly_types:
                if anomaly['anomaly_type'] == attack_type:
                    result = anomaly.copy()
                    result["similarity_score"] = float(similarity)
                    result["profile_based"] = True
                    results.append(result)
                    break

        return results

    def _fallback_retrieval(self, top_k: int):
        """Fallback if no profiles are available"""
        return self.anomaly_types[:min(top_k, len(self.anomaly_types))]

    def generate_explanation(self, feature_vector: np.array, prediction: int, confidence: float,
                             retrieved_anomalies: List[Dict]) -> str:
        """Generate natural language explanation using local deepseek model"""

        # Prepare context from retrieved anomalies
        context = "\n".join([
            f"Anomaly Type: {anom['anomaly_type']}\n"
            f"Description: {anom['description']}\n"
            f"Key Features: {', '.join(anom['features'][:5])}\n"
            f"Remediation: {anom['remediation']}\n"
            f"Severity: {anom['severity']}\n"
            f"Similarity Score: {anom.get('similarity_score', 0):.3f}\n"
            for anom in retrieved_anomalies
        ])

        prompt = f"""
### Role: You are a network security expert analyzing anomaly detection results.

### Context from knowledge base:
{context}

### Current detection results:
- Prediction: {'ATTACK' if prediction == 1 else 'NORMAL'}
- Confidence: {confidence:.3f}
- Top features: {', '.join([f'feature_{i}={v:.3f}' for i, v in enumerate(feature_vector[:5])])}

### Task: Generate a concise security analysis including:
1. Likely anomaly type based on similarity scores
2. Key features that contributed to this decision
3. Recommended immediate actions
4. Severity assessment and business impact

### Response Format:
Provide a structured analysis in 3-4 paragraphs.

### Analysis:
"""

        try:
            explanation = self._generate_text(prompt)
            return explanation
        except Exception as e:
            print(f"Explanation generation failed: {e}")
            return self._generate_fallback_explanation(prediction, confidence,
                                                       retrieved_anomalies[0] if retrieved_anomalies else {})

    def _generate_fallback_explanation(self, prediction: int, confidence: float, anomaly: Dict) -> str:
        """Fallback explanation without LLM"""
        if prediction == 0:
            return f"Normal network traffic detected with confidence {confidence:.3f}. No significant anomalies found in the analyzed patterns."

        return (
            f"Security anomaly detected with confidence {confidence:.3f}. "
            f"This pattern most closely resembles {anomaly.get('anomaly_type', 'unknown')} activity. "
            f"Key indicators include: {anomaly.get('description', 'suspicious network behavior')}. "
            f"Immediate action: {anomaly.get('remediation', 'investigate and contain')}. "
            f"Severity level: {anomaly.get('severity', 'requires assessment')}."
        )

    def explain_anomaly(self, feature_vector: np.array, prediction: int, confidence: float) -> Dict:
        """Complete RAG explanation pipeline"""
        # Retrieve similar anomalies
        retrieved = self.retrieve_similar_anomalies(feature_vector, top_k=5)

        # Generate explanation
        explanation = self.generate_explanation(feature_vector, prediction, confidence, retrieved)

        return {
            "prediction": "Attack" if prediction == 1 else "Normal",
            "confidence": float(confidence),
            "similar_anomalies": retrieved,
            "explanation": explanation,
            "key_features": self._extract_key_features(feature_vector, top_n=5)
        }

    def _extract_key_features(self, feature_vector: np.array, top_n: int = 5) -> List[Dict]:
        """Extract top contributing features"""
        abs_features = np.abs(feature_vector)
        top_indices = abs_features.argsort()[-top_n:][::-1]

        return [
            {
                "feature_index": int(idx),
                "feature_value": float(feature_vector[idx]),
                "contribution": float(abs_features[idx]),
                "interpretation": self._interpret_feature(idx, feature_vector[idx])
            }
            for idx in top_indices
        ]

    def _interpret_feature(self, index: int, value: float) -> str:
        """Provide interpretation using JSON knowledge base"""
        index_str = str(index)

        if index_str in self.feature_interpretations:
            feat_info = self.feature_interpretations[index_str]
            name = feat_info.get("name", f"Feature {index}")
            description = feat_info.get("description", "")
            attack_context = feat_info.get("attack_context", "Network behavior indicator")

            # Add value-based interpretation
            abs_value = abs(value)
            if abs_value > 2.0:
                magnitude = "extremely "
                severity = "critical anomaly"
            elif abs_value > 1.5:
                magnitude = "highly "
                severity = "major anomaly"
            elif abs_value > 1.0:
                magnitude = "very "
                severity = "significant anomaly"
            elif abs_value > 0.5:
                magnitude = "moderately "
                severity = "noticeable anomaly"
            else:
                magnitude = "slightly "
                severity = "minor deviation"

            direction = "above" if value > 0 else "below"

            return f"{name} ({magnitude}{direction} normal - {severity}) - {attack_context}"

        else:
            # Fallback for unknown features
            abs_value = abs(value)
            if abs_value > 1.0:
                severity = "significant"
            elif abs_value > 0.5:
                severity = "moderate"
            else:
                severity = "minor"

            direction = "above" if value > 0 else "below"

            return f"Feature {index} ({severity} {direction} normal) - General network behavior indicator"