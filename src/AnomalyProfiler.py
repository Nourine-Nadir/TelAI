import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder


class AnomalyFeatureProfiler:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.feature_profiles = {}
        self.feature_means = {}
        self.feature_stds = {}

    def compute_attack_profiles(self):
        """Compute feature profiles for each attack type from labeled data"""
        print("Loading data and computing attack profiles...")

        # Load the data
        df = pd.read_csv(self.data_path)

        # Check if we have the necessary columns
        if 'attack_cat' not in df.columns or 'label' not in df.columns:
            raise ValueError("Data must contain 'attack_cat' and 'label' columns")

        # Preprocess the data (similar to your training preprocessing)
        df = df.drop(['id'], axis=1, errors='ignore')

        # Encode categorical features
        categorical_cols = ["proto", "service", "state"]
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

        # Separate features and labels
        feature_columns = [col for col in df.columns if col not in ['attack_cat', 'label']]
        X = df[feature_columns].values
        y_attack = df['attack_cat'].values
        y_binary = df['label'].values

        # Compute overall statistics for normalization
        self.feature_means = np.mean(X, axis=0)
        self.feature_stds = np.std(X, axis=0) + 1e-8

        # Normalize the features
        X_normalized = (X - self.feature_means) / self.feature_stds

        # Compute profiles for each attack type
        attack_types = np.unique(y_attack)

        for attack_type in attack_types:
            if attack_type == 'Normal':  # Skip normal traffic
                continue

            # Get samples for this attack type
            attack_mask = (y_attack == attack_type)
            X_attack = X_normalized[attack_mask]

            if len(X_attack) > 0:
                # Compute mean feature values for this attack
                attack_profile = np.mean(np.abs(X_attack), axis=0)

                # Normalize to create a probability distribution
                attack_profile = attack_profile / (np.sum(attack_profile) + 1e-8)

                self.feature_profiles[attack_type] = {
                    'profile': attack_profile.tolist(),
                    'sample_count': len(X_attack),
                    'mean_abnormality': np.mean(attack_profile),
                    'max_abnormality': np.max(attack_profile)
                }

                print(f"Computed profile for {attack_type}: {len(X_attack)} samples")

        # Also compute normal traffic profile for comparison
        normal_mask = (y_attack == 'Normal')
        if np.any(normal_mask):
            X_normal = X_normalized[normal_mask]
            normal_profile = np.mean(np.abs(X_normal), axis=0)
            normal_profile = normal_profile / (np.sum(normal_profile) + 1e-8)

            self.feature_profiles['Normal'] = {
                'profile': normal_profile.tolist(),
                'sample_count': len(X_normal),
                'mean_abnormality': np.mean(normal_profile),
                'max_abnormality': np.max(normal_profile)
            }

        return self.feature_profiles

    def save_profiles(self, output_path: str):
        """Save computed profiles to JSON file"""
        profiles_data = {
            'feature_means': self.feature_means.tolist(),
            'feature_stds': self.feature_stds.tolist(),
            'feature_profiles': self.feature_profiles,
            'feature_names': [f"feature_{i}" for i in range(len(self.feature_means))]
        }

        with open(output_path, 'w') as f:
            json.dump(profiles_data, f, indent=2)

        print(f"Profiles saved to {output_path}")

    def load_profiles(self, input_path: str):
        """Load precomputed profiles from JSON file"""
        with open(input_path, 'r') as f:
            profiles_data = json.load(f)

        self.feature_means = np.array(profiles_data['feature_means'])
        self.feature_stds = np.array(profiles_data['feature_stds'])
        self.feature_profiles = profiles_data['feature_profiles']

        print(f"Loaded profiles for {len(self.feature_profiles)} attack types")
        return self.feature_profiles


# Usage example
if __name__ == "__main__":
    profiler = AnomalyFeatureProfiler("../Data/UNSW-NB15/UNSW_NB15_training-set.csv")
    profiles = profiler.compute_attack_profiles()
    profiler.save_profiles("attack_profiles.json")