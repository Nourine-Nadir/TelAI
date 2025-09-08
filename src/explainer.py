import numpy as np
import pandas as pd
import torch
import json
import ollama
from typing import List, Dict, Any
import time
from utils import preprocess_unsw


class AnomalyRAGExplainer:
    def __init__(self, knowledge_base_path: str = None):
        # Use local Ollama models instead of HF models
        self.embedding_model_name = "nomic-embed-text"
        self.generator_model_name = "deepseek-r1:14b"

        # Load or create knowledge base
        self.knowledge_base = self._load_knowledge_base(knowledge_base_path)
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
                    return json.load(f)
            except:
                print("Could not load knowledge base, using UNSW-NB15 default")

        # UNSW-NB15 attack categories knowledge base
        return [
            {
                "anomaly_type": "Fuzzers",
                "description": "Fuzzing attacks that send malformed or unexpected input to crash services or find vulnerabilities",
                "features": ["irregular packet sizes", "multiple protocol violations", "unusual service requests",
                             "high error rates"],
                "remediation": "Input validation, protocol compliance checking, intrusion prevention systems",
                "severity": "Medium-High",
                "examples": ["Buffer overflow attempts", "Protocol fuzzing", "Input manipulation"]
            },
            {
                "anomaly_type": "Analysis",
                "description": "Traffic analysis and network mapping activities to gather intelligence about network structure",
                "features": ["multiple port connections", "various protocol usage", "low traffic volume",
                             "systematic scanning patterns"],
                "remediation": "Network segmentation, port security, traffic monitoring for reconnaissance patterns",
                "severity": "Medium",
                "examples": ["Network mapping", "Service discovery", "Topology analysis"]
            },
            {
                "anomaly_type": "Backdoors",
                "description": "Unauthorized access channels created to bypass normal authentication mechanisms",
                "features": ["unusual outbound connections", "encrypted traffic to unknown destinations",
                             "persistent connections", "off-hours activity"],
                "remediation": "Access control reviews, connection monitoring, malware scanning, firewall rules audit",
                "severity": "Critical",
                "examples": ["Trojan communications", "C2 channel activity", "Covert tunnels"]
            },
            {
                "anomaly_type": "DoS",
                "description": "Denial of Service attacks aimed at making resources unavailable to legitimate users",
                "features": ["extremely high packet rates", "multiple source IPs", "small packet sizes",
                             "resource exhaustion patterns"],
                "remediation": "Rate limiting, traffic filtering, DDoS protection services, resource monitoring",
                "severity": "High-Critical",
                "examples": ["SYN floods", "UDP floods", "ICMP floods", "HTTP floods"]
            },
            {
                "anomaly_type": "Exploits",
                "description": "Attacks that exploit specific vulnerabilities in software or systems",
                "features": ["targeted payload delivery", "specific service targeting",
                             "vulnerability-specific patterns", "privilege escalation attempts"],
                "remediation": "Patch management, vulnerability scanning, application firewall, least privilege principles",
                "severity": "High",
                "examples": ["Remote code execution", "SQL injection", "XSS attacks", "Privilege escalation"]
            },
            {
                "anomaly_type": "Generic",
                "description": "General attacks that don't fit specific categories but show clear malicious intent",
                "features": ["multiple attack signatures", "mixed malicious patterns", "general suspicious behavior",
                             "policy violations"],
                "remediation": "General security hardening, behavior analysis, comprehensive logging, security awareness",
                "severity": "Medium-High",
                "examples": ["Multi-vector attacks", "General malware activity", "Policy violations"]
            },
            {
                "anomaly_type": "Reconnaissance",
                "description": "Information gathering activities to identify potential targets and vulnerabilities",
                "features": ["systematic port scanning", "service enumeration", "network probing",
                             "low-and-slow traffic patterns"],
                "remediation": "Intrusion detection systems, honeypots, network monitoring, access logging",
                "severity": "Low-Medium",
                "examples": ["Port scanning", "Service discovery", "Network mapping", "Vulnerability scanning"]
            },
            {
                "anomaly_type": "Shellcode",
                "description": "Malicious code execution attempts through injected shellcode payloads",
                "features": ["executable code in packets", "memory corruption patterns", "return-oriented programming",
                             "payload obfuscation"],
                "remediation": "Memory protection, executable space protection, code signing, behavior monitoring",
                "severity": "High",
                "examples": ["Buffer overflow exploits", "Code injection", "ROP chain attacks"]
            },
            {
                "anomaly_type": "Worms",
                "description": "Self-replicating malware that spreads through network vulnerabilities",
                "features": ["rapid network propagation", "multiple infection attempts", "scanning behavior",
                             "automatic replication"],
                "remediation": "Patch management, network segmentation, behavior analysis, containment procedures",
                "severity": "Critical",
                "examples": ["Network worms", "Auto-replicating malware", "Lateral movement"]
            },
            {
                "anomaly_type": "Normal",
                "description": "Legitimate network traffic without malicious intent",
                "features": ["consistent patterns", "expected protocols", "normal traffic volumes",
                             "authorized services"],
                "remediation": "Continue normal monitoring, maintain security baselines",
                "severity": "None",
                "examples": ["Web browsing", "Email traffic", "File transfers", "Database queries"]
            }
        ]

    def _embed_knowledge_base(self):
        """Create embeddings for knowledge base using local model"""
        print("Embedding knowledge base with nomic-embed-text...")
        self.embeddings = []

        for item in self.knowledge_base:
            text = f"{item['anomaly_type']}: {item['description']}. Features: {', '.join(item['features'])}"
            embedding = self._embed_text(text)
            self.embeddings.append(embedding)
            time.sleep(0.1)  # Small delay to avoid overwhelming Ollama

        self.embeddings = np.array(self.embeddings)
        print("Knowledge base embedding completed!")

    def retrieve_similar_anomalies(self, feature_vector: np.array, top_k: int = 3) -> List[Dict]:
        """Retrieve most similar anomalies from knowledge base"""
        if self.embeddings is None or len(self.embeddings) == 0:
            return self.knowledge_base[:top_k]

        # Convert features to text description for embedding
        feature_text = f"Network traffic with features: {', '.join([f'feature_{i}={v:.3f}' for i, v in enumerate(feature_vector[:10])])}"  # Use first 10 features for brevity

        query_embedding = np.array(self._embed_text(feature_text))

        # Calculate similarities
        similarities = np.dot(self.embeddings, query_embedding) / (
                np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Get top-k most similar
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            result = self.knowledge_base[idx].copy()
            result["similarity_score"] = float(similarities[idx])
            results.append(result)

        return results

    def generate_explanation(self, feature_vector: np.array, prediction: int, confidence: float,
                             retrieved_anomalies: List[Dict]) -> str:
        """Generate natural language explanation using local deepseek model"""

        # Prepare context from retrieved anomalies
        context = "\n".join([
            f"Anomaly Type: {anom['anomaly_type']}\n"
            f"Description: {anom['description']}\n"
            f"Key Features: {', '.join(anom['features'][:3])}\n"
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
        retrieved = self.retrieve_similar_anomalies(feature_vector)

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
        """Provide interpretation of UNSW-NB15 feature values with attack context"""

        # UNSW-NB15 feature interpretations
        interpretations = {
            # Basic connection features
            0: ("Connection duration", "Duration of the network connection",
                "Long durations may indicate persistent C2 connections or data exfiltration"),
            1: ("Protocol type", "Network protocol used (TCP/UDP/ICMP/etc)",
                "Unusual protocols may indicate evasion techniques"),
            2: ("Service type", "Network service (HTTP/FTP/DNS/etc)",
                "Unexpected services may indicate service scanning or exploitation"),
            3: ("Connection state", "State of the connection (ESTABLISHED/CONNECTING/etc)",
                "Abnormal states may indicate connection manipulation"),

            # Packet statistics
            4: ("Source packets", "Number of packets from source to destination",
                "High counts may indicate DoS flooding or scanning"),
            5: ("Destination packets", "Number of packets from destination to source",
                "Low counts during attacks may indicate one-way traffic"),
            6: ("Source bytes", "Number of data bytes from source to destination",
                "Large volumes may indicate data exfiltration or DoS attacks"),
            7: ("Destination bytes", "Number of data bytes from destination to source",
                "Small volumes may indicate reconnaissance activity"),
            8: ("Packet rate", "Rate of packets per second", "Extremely high rates indicate DoS attacks"),

            # Time-to-Live (TTL) values
            9: ("Source TTL", "Time-to-live value of source IP packets",
                "TTL manipulation may indicate OS fingerprinting or evasion"),
            10: ("Destination TTL", "Time-to-live value of destination IP packets",
                 "Abnormal TTL values may suggest spoofing"),

            # Load and throughput
            11: ("Source load", "Source bits per second",
                 "Abnormal loads may indicate various attacks including brute force"),
            12: ("Destination load", "Destination bits per second",
                 "Unexpected loads may indicate resource targeting or exploitation"),

            # Packet loss
            13: ("Source packet loss", "Number of retransmitted or lost source packets",
                 "High loss may indicate network congestion or attack"),
            14: ("Destination packet loss", "Number of retransmitted or lost destination packets",
                 "Loss patterns may indicate targeted attacks"),

            # Inter-packet timing
            15: ("Source inter-packet time", "Time between source packets (ms)",
                 "Short times may indicate flooding, regular intervals may indicate C2"),
            16: ("Destination inter-packet time", "Time between destination packets (ms)",
                 "Irregular times may indicate malicious activity"),
            17: ("Source jitter", "Variation in source inter-packet times",
                 "High jitter may indicate fuzzing or malformed packets"),
            18: ("Destination jitter", "Variation in destination inter-packet times",
                 "Irregular responses may indicate service stress"),

            # TCP window sizes
            19: ("Source TCP window", "TCP window size of source",
                 "Abnormal window sizes may indicate exploitation attempts"),
            20: ("Source TCP base", "TCP base sequence number of source", "Sequence anomalies may indicate hijacking"),
            21: (
            "Destination TCP base", "TCP base sequence number of destination", "Sequence issues may indicate attacks"),
            22: ("Destination TCP window", "TCP window size of destination",
                 "Window size manipulation may indicate attacks"),

            # TCP timing metrics
            23: ("TCP RTT", "TCP round-trip time", "Abnormal RTT may indicate network manipulation or congestion"),
            24: ("SYN-ACK time", "Time between SYN and SYN-ACK packets",
                 "Long times may indicate half-open connections (DoS)"),
            25: (
            "ACK data time", "Time between ACK and data packets", "Timing anomalies may indicate protocol violations"),

            # Mean values
            26: (
            "Source mean", "Mean of source packet sizes", "Unusual mean sizes may indicate specific attack payloads"),
            27: (
            "Destination mean", "Mean of destination packet sizes", "Packet size patterns may indicate attack types"),

            # Transaction depth
            28: ("Transaction depth", "Depth of transaction (for HTTP/FTP)",
                 "Abnormal depths may indicate HTTP exploits or web attacks"),

            # Response body
            29: ("Response body length", "Length of response body content",
                 "Unexpected lengths may indicate successful exploits or data leakage"),

            # Connection tracking features
            30: ("Service-source count", "Number of connections for same service and source",
                 "High counts suggest port scanning (Reconnaissance)"),
            31: ("State-TTL count", "Number of connections for same state and TTL",
                 "Patterns may indicate scanning or fingerprinting"),
            32: ("Destination count", "Number of connections for same destination",
                 "Multiple connections to same target suggest scanning or DoS"),
            33: ("Source-dest port count", "Number of connections for same source and destination port",
                 "Patterns may indicate specific service attacks"),
            34: ("Dest-source port count", "Number of connections for same destination and source port",
                 "Connection patterns may reveal attack vectors"),
            35: ("Destination-source count", "Number of connections for same destination and source",
                 "Relationship patterns may indicate compromised systems"),

            # FTP features
            36: ("FTP login", "Whether FTP login was attempted (1/0)", "Multiple attempts suggest brute force attacks"),
            37: ("FTP commands", "Number of FTP commands",
                 "Unusual commands may indicate FTP exploits or unauthorized access"),

            # HTTP features
            38: ("HTTP methods", "Number of HTTP methods", "Unusual methods may indicate web application attacks"),

            # Additional connection tracking
            39: ("Source count", "Number of connections for same source",
                 "High counts from single source suggest infected host or scanning"),
            40: ("Service-destination count", "Number of connections for same service and destination",
                 "High counts may indicate service-specific attacks"),
            41: ("Same IPs and ports", "Whether source and destination IPs/ports are same (1/0)",
                 "Reflection attacks or self-connections may indicate malware")
        }

        # Get the interpretation or default
        if index in interpretations:
            feature_name, description, attack_context = interpretations[index]

            # Add value-based interpretation with attack context
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

            return f"{feature_name} ({magnitude}{direction} normal - {severity}) - {attack_context}"

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

# Test the updated implementation
if __name__ == "__main__":
    # Initialize explainer with local models
    explainer = AnomalyRAGExplainer()

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
