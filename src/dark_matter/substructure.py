"""
Dark Matter Substructure Detection Module

Implements subhalo population modeling and ML-based detection of
dark matter substructure from gravitational lensing flux ratio anomalies.

Physics
-------
Subhalos cause flux ratio anomalies in multiply-imaged quasars.
The subhalo mass function follows: dN/dM ∝ M^(-1.9)

Machine Learning
----------------
Binary classifier to detect substructure presence from flux ratios.

References
----------
- Dalal & Kochanek (2002) ApJ 572, 25 - Flux ratio anomalies
- Vegetti et al. (2012) Nature 481, 341 - Direct detection
- Hezaveh et al. (2016) ApJ 823, 37 - Neural networks for substructure
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass

@dataclass
class Subhalo:
    """Single subhalo properties."""
    mass: float  # Solar masses
    position: Tuple[float, float]  # (x, y) in arcsec
    concentration: float = 10.0


class SubhaloPopulation:
    """
    Generate subhalo population following mass function.
    
    Parameters
    ----------
    mass_min : float
        Minimum subhalo mass (Msun)
    mass_max : float
        Maximum subhalo mass (Msun)
    alpha : float
        Mass function slope (default: -1.9)
    fov : float
        Field of view in arcsec
    """
    
    def __init__(
        self,
        mass_min: float = 1e6,
        mass_max: float = 1e10,
        alpha: float = -1.9,
        fov: float = 10.0
    ):
        self.mass_min = mass_min
        self.mass_max = mass_max
        self.alpha = alpha
        self.fov = fov
    
    def generate_population(
        self,
        total_mass_fraction: float = 0.01,
        host_mass: float = 1e13
    ) -> List[Subhalo]:
        """
        Generate subhalo population.
        
        Parameters
        ----------
        total_mass_fraction : float
            Fraction of host mass in subhalos
        host_mass : float
            Host halo mass (Msun)
        
        Returns
        -------
        subhalos : list
            List of Subhalo objects
        """
        # Total subhalo mass
        total_subhalo_mass = host_mass * total_mass_fraction
        
        # Generate masses from power-law
        n_halos = int(1000 * (total_mass_fraction / 0.01))
        
        # Power-law sampling
        u = np.random.uniform(0, 1, n_halos)
        if self.alpha != -1:
            masses = ((self.mass_max**(self.alpha+1) - self.mass_min**(self.alpha+1)) * u 
                     + self.mass_min**(self.alpha+1))**(1/(self.alpha+1))
        else:
            masses = self.mass_min * (self.mass_max/self.mass_min)**u
        
        # Normalize to total mass
        masses = masses * (total_subhalo_mass / np.sum(masses))
        
        # Generate positions (uniform in FOV)
        positions = [
            (np.random.uniform(-self.fov/2, self.fov/2),
             np.random.uniform(-self.fov/2, self.fov/2))
            for _ in range(len(masses))
        ]
        
        # Create subhalos
        subhalos = [
            Subhalo(mass=m, position=pos, concentration=10.0)
            for m, pos in zip(masses, positions)
        ]
        
        return subhalos
    
    def mass_function_stats(self, subhalos: List[Subhalo]) -> Dict:
        """Get statistics of subhalo population."""
        masses = np.array([s.mass for s in subhalos])
        return {
            'n_subhalos': len(subhalos),
            'total_mass': np.sum(masses),
            'mean_mass': np.mean(masses),
            'median_mass': np.median(masses),
            'mass_range': (np.min(masses), np.max(masses))
        }


class SubstructureDetector:
    """
    ML-based substructure detection from flux ratio anomalies.
    
    Parameters
    ----------
    model_type : str
        'random_forest' or 'neural_net'
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.trained = False
        self.model = None
    
    def extract_features(
        self,
        flux_ratios: np.ndarray,
        image_positions: np.ndarray
    ) -> np.ndarray:
        """
        Extract features for ML classification.
        
        Parameters
        ----------
        flux_ratios : np.ndarray
            Observed flux ratios between images
        image_positions : np.ndarray
            Image positions (N, 2)
        
        Returns
        -------
        features : np.ndarray
            Feature vector
        """
        features = []
        
        # Flux ratio statistics
        features.append(np.mean(flux_ratios))
        features.append(np.std(flux_ratios))
        features.append(np.max(flux_ratios) / np.min(flux_ratios))
        
        # Position statistics
        features.append(np.mean(np.linalg.norm(image_positions, axis=1)))
        features.append(np.std(np.linalg.norm(image_positions, axis=1)))
        
        # Flux ratio anomalies (deviation from smooth model)
        predicted_smooth = self._smooth_model_prediction(image_positions)
        anomaly = np.abs(flux_ratios - predicted_smooth)
        features.append(np.mean(anomaly))
        features.append(np.max(anomaly))
        
        return np.array(features)
    
    def _smooth_model_prediction(self, positions: np.ndarray) -> np.ndarray:
        """Predict smooth model flux ratios (placeholder)."""
        # In real implementation, use smooth lens model
        return np.ones(len(positions))
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> None:
        """
        Train substructure detector.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training features (N, n_features)
        y_train : np.ndarray
            Training labels (N,) - 1=substructure, 0=smooth
        """
        if self.model_type == 'random_forest':
            try:
                from sklearn.ensemble import RandomForestClassifier
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            except ImportError:
                raise ImportError("scikit-learn is required for Random Forest. Please install it.")
        elif self.model_type == 'neural_net':
            try:
                from sklearn.neural_network import MLPClassifier
                self.model = MLPClassifier(hidden_layer_sizes=(50, 50), random_state=42)
            except ImportError:
                raise ImportError("scikit-learn is required for MLPClassifier. Please install it.")
        
        self.model.fit(X_train, y_train)
        self.trained = True
    
    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict substructure presence.
        
        Parameters
        ----------
        X_test : np.ndarray
            Test features
        
        Returns
        -------
        predictions : np.ndarray
            Binary predictions
        probabilities : np.ndarray
            Prediction probabilities
        """
        if not self.trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)[:, 1]
        
        return predictions, probabilities
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """
        Evaluate detector performance.
        
        Returns
        -------
        metrics : dict
            Accuracy, precision, recall, F1 score
        """
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        except ImportError:
            raise ImportError("scikit-learn is required for evaluation.")
            
        predictions, _ = self.predict(X_test)
        
        return {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions),
            'recall': recall_score(y_test, predictions),
            'f1_score': f1_score(y_test, predictions)
        }


def generate_training_data(
    n_samples: int = 1000,
    substructure_fraction: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data for substructure detection.
    
    Parameters
    ----------
    n_samples : int
        Number of training samples
    substructure_fraction : float
        Fraction with substructure
    
    Returns
    -------
    X : np.ndarray
        Features (n_samples, n_features)
    y : np.ndarray
        Labels (n_samples,)
    """
    n_substructure = int(n_samples * substructure_fraction)
    n_smooth = n_samples - n_substructure
    
    X = []
    y = []
    
    # Generate samples with substructure
    for _ in range(n_substructure):
        # Add anomalies
        flux_ratios = np.random.uniform(0.5, 2.0, 4) + np.random.normal(0, 0.3, 4)
        positions = np.random.uniform(-5, 5, (4, 2))
        
        detector = SubstructureDetector()
        features = detector.extract_features(flux_ratios, positions)
        
        X.append(features)
        y.append(1)  # Has substructure
    
    # Generate smooth samples
    for _ in range(n_smooth):
        flux_ratios = np.random.uniform(0.8, 1.2, 4)
        positions = np.random.uniform(-5, 5, (4, 2))
        
        detector = SubstructureDetector()
        features = detector.extract_features(flux_ratios, positions)
        
        X.append(features)
        y.append(0)  # Smooth
    
    return np.array(X), np.array(y)


if __name__ == "__main__":
    print("Dark Matter Substructure Detection Module")
    print("=" * 70)
    
    # Test subhalo population
    print("\n1. Generating subhalo population...")
    pop = SubhaloPopulation()
    subhalos = pop.generate_population(total_mass_fraction=0.01, host_mass=1e13)
    stats = pop.mass_function_stats(subhalos)
    
    print(f"Generated {stats['n_subhalos']} subhalos")
    print(f"Total mass: {stats['total_mass']:.2e} Msun")
    print(f"Mass range: {stats['mass_range'][0]:.2e} - {stats['mass_range'][1]:.2e} Msun")
    
    print("\n[SUCCESS] Substructure module loaded successfully")
