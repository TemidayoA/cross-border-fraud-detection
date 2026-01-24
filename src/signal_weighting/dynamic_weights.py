"""
Dynamic Signal Weighting for Corridor-Aware Fraud Detection

This module implements the core logic for adjusting fraud signal weights
based on payment corridor characteristics. Instead of one-size-fits-all
thresholds, each corridor gets optimised feature importance.

Author: Temidayo
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class CorridorProfile:
    """Statistical profile for a payment corridor."""
    corridor_id: str
    corridor_name: str
    
    # Amount statistics
    amount_median: float
    amount_p95: float
    amount_p99: float
    
    # Velocity statistics  
    velocity_median: float
    velocity_p95: float
    
    # Temporal patterns
    peak_hours: List[int]
    peak_days: List[int]
    
    # Risk metrics
    historical_fraud_rate: float
    fraud_amount_multiplier: float = 1.5  # Fraud amounts typically X times higher
    
    # Corridor tier (1=low risk, 4=very high risk)
    risk_tier: int = 2


@dataclass
class BaseWeights:
    """Base feature weights before corridor adjustment."""
    amount_deviation: float = 0.25
    velocity: float = 0.25
    temporal_anomaly: float = 0.10
    sender_maturity: float = 0.20
    beneficiary_novelty: float = 0.20
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'amount_deviation': self.amount_deviation,
            'velocity': self.velocity,
            'temporal_anomaly': self.temporal_anomaly,
            'sender_maturity': self.sender_maturity,
            'beneficiary_novelty': self.beneficiary_novelty,
        }


@dataclass 
class CorridorMultipliers:
    """
    Multipliers that adjust base weights for specific corridors.
    
    Values > 1.0 increase importance of that signal for this corridor.
    Values < 1.0 decrease importance.
    """
    amount_deviation: float = 1.0
    velocity: float = 1.0
    temporal_anomaly: float = 1.0
    sender_maturity: float = 1.0
    beneficiary_novelty: float = 1.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'amount_deviation': self.amount_deviation,
            'velocity': self.velocity,
            'temporal_anomaly': self.temporal_anomaly,
            'sender_maturity': self.sender_maturity,
            'beneficiary_novelty': self.beneficiary_novelty,
        }


# Pre-defined corridor multipliers based on observed patterns
CORRIDOR_MULTIPLIERS = {
    'GBP_NGN': CorridorMultipliers(
        amount_deviation=1.2,    # Amount more predictive (fraud tends higher)
        velocity=0.8,            # Velocity less predictive (high legitimate volume)
        temporal_anomaly=0.6,    # Timing less predictive (24/7 remittance culture)
        sender_maturity=1.0,     # Standard
        beneficiary_novelty=1.5, # New beneficiary highly suspicious
    ),
    'GBP_PLN': CorridorMultipliers(
        amount_deviation=0.9,    # Amount less predictive (narrow range)
        velocity=1.4,            # Velocity highly predictive (regular monthly pattern)
        temporal_anomaly=1.2,    # Timing more predictive (weekday payroll pattern)
        sender_maturity=0.8,     # Less important (established migrant worker base)
        beneficiary_novelty=0.7, # Less suspicious (workers may change recipients)
    ),
    'GBP_INR': CorridorMultipliers(
        amount_deviation=1.1,    # Slightly elevated (high-value legitimate transactions)
        velocity=1.0,            # Standard
        temporal_anomaly=0.9,    # Slightly lower
        sender_maturity=1.2,     # New accounts more suspicious
        beneficiary_novelty=1.3, # Elevated
    ),
    'GBP_PHP': CorridorMultipliers(
        amount_deviation=1.0,    # Standard
        velocity=0.7,            # Lower (high legitimate frequency)
        temporal_anomaly=0.8,    # Lower (unusual hours normal due to timezone)
        sender_maturity=1.1,     # Slightly elevated
        beneficiary_novelty=1.2, # Elevated
    ),
}


class DynamicWeightCalculator:
    """
    Calculates corridor-adjusted feature weights for fraud scoring.
    
    The core innovation: instead of fixed weights, we adjust feature
    importance based on what's actually predictive for each corridor.
    """
    
    def __init__(
        self,
        base_weights: Optional[BaseWeights] = None,
        corridor_multipliers: Optional[Dict[str, CorridorMultipliers]] = None
    ):
        self.base_weights = base_weights or BaseWeights()
        self.corridor_multipliers = corridor_multipliers or CORRIDOR_MULTIPLIERS
        
        # Cache for computed weights
        self._weight_cache: Dict[str, Dict[str, float]] = {}
    
    def get_adjusted_weights(self, corridor_id: str) -> Dict[str, float]:
        """
        Calculate adjusted weights for a specific corridor.
        
        Returns normalised weights that sum to 1.0.
        """
        # Check cache first
        if corridor_id in self._weight_cache:
            return self._weight_cache[corridor_id]
        
        # Get multipliers (default to 1.0 for unknown corridors)
        multipliers = self.corridor_multipliers.get(
            corridor_id, 
            CorridorMultipliers()
        )
        
        base = self.base_weights.to_dict()
        mults = multipliers.to_dict()
        
        # Apply multipliers
        adjusted = {
            feature: base[feature] * mults[feature]
            for feature in base.keys()
        }
        
        # Normalise to sum to 1.0
        total = sum(adjusted.values())
        normalised = {
            feature: weight / total
            for feature, weight in adjusted.items()
        }
        
        # Cache and return
        self._weight_cache[corridor_id] = normalised
        return normalised
    
    def clear_cache(self):
        """Clear the weight cache (call after updating multipliers)."""
        self._weight_cache.clear()
    
    def get_weight_comparison(self) -> Dict[str, Dict[str, float]]:
        """
        Return weight comparison across all configured corridors.
        Useful for understanding how weights differ.
        """
        comparison = {'base': self.base_weights.to_dict()}
        
        for corridor_id in self.corridor_multipliers.keys():
            comparison[corridor_id] = self.get_adjusted_weights(corridor_id)
        
        return comparison


class FraudScorer:
    """
    Complete fraud scoring engine with dynamic weight adjustment.
    
    This is the main class that combines:
    - Feature calculation
    - Dynamic weight adjustment
    - Infrastructure health checks
    - Final score computation
    """
    
    def __init__(
        self,
        weight_calculator: Optional[DynamicWeightCalculator] = None,
        corridor_profiles: Optional[Dict[str, CorridorProfile]] = None,
    ):
        self.weight_calculator = weight_calculator or DynamicWeightCalculator()
        self.corridor_profiles = corridor_profiles or {}
        
        # Infrastructure status (would connect to real monitoring in production)
        self.infrastructure_status: Dict[str, float] = defaultdict(lambda: 1.0)
    
    def calculate_fraud_score(
        self,
        features: Dict[str, float],
        corridor_id: str,
        apply_infrastructure_adjustment: bool = True
    ) -> Dict:
        """
        Calculate final fraud score for a transaction.
        
        Args:
            features: Dict of feature scores (each 0-1 range)
            corridor_id: Payment corridor identifier
            apply_infrastructure_adjustment: Whether to adjust for infra issues
            
        Returns:
            Dict containing score, breakdown, and explanation
        """
        # Get corridor-adjusted weights
        weights = self.weight_calculator.get_adjusted_weights(corridor_id)
        
        # Calculate weighted sum
        raw_score = sum(
            features.get(feature, 0) * weight
            for feature, weight in weights.items()
        )
        
        # Apply infrastructure adjustment if enabled
        infra_adjustment = 0.0
        if apply_infrastructure_adjustment:
            infra_health = self.infrastructure_status.get(corridor_id, 1.0)
            if infra_health < 0.7:  # Degraded infrastructure
                # Reduce score - some "suspicious" activity may be retries
                infra_adjustment = -0.15 * (1 - infra_health)
        
        adjusted_score = max(0, min(1, raw_score + infra_adjustment))
        
        # Get corridor baseline offset (higher-risk corridors get small boost)
        baseline_offset = self._get_corridor_baseline(corridor_id)
        final_score = max(0, min(1, adjusted_score + baseline_offset))
        
        # Generate explanation
        explanation = self._generate_explanation(
            features, weights, final_score, corridor_id
        )
        
        return {
            'score': round(final_score, 4),
            'raw_score': round(raw_score, 4),
            'decision': self._make_decision(final_score),
            'features': features,
            'weights': weights,
            'adjustments': {
                'infrastructure': round(infra_adjustment, 4),
                'baseline': round(baseline_offset, 4),
            },
            'explanation': explanation,
        }
    
    def _get_corridor_baseline(self, corridor_id: str) -> float:
        """
        Get baseline score offset for corridor.
        Higher-risk corridors get a small positive offset.
        """
        profile = self.corridor_profiles.get(corridor_id)
        if not profile:
            return 0.0
        
        # Map risk tier to baseline offset
        tier_offsets = {
            1: -0.05,  # Low risk: slight reduction
            2: 0.0,    # Medium: no change
            3: 0.05,   # High: slight increase  
            4: 0.10,   # Very high: moderate increase
        }
        
        return tier_offsets.get(profile.risk_tier, 0.0)
    
    def _make_decision(self, score: float) -> str:
        """Convert score to decision."""
        if score < 0.3:
            return 'APPROVE'
        elif score < 0.6:
            return 'REVIEW'
        else:
            return 'BLOCK'
    
    def _generate_explanation(
        self,
        features: Dict[str, float],
        weights: Dict[str, float],
        final_score: float,
        corridor_id: str
    ) -> Dict:
        """Generate human-readable explanation for the decision."""
        
        # Calculate weighted contribution of each feature
        contributions = {
            feature: features.get(feature, 0) * weights[feature]
            for feature in weights.keys()
        }
        
        # Sort by contribution
        sorted_contrib = sorted(
            contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Primary factors (top contributors > 0.05)
        primary = [
            f"{feat.replace('_', ' ').title()}: {features.get(feat, 0):.2f}"
            for feat, contrib in sorted_contrib
            if contrib > 0.05
        ]
        
        # Mitigating factors (low scores on weighted features)
        mitigating = [
            f"{feat.replace('_', ' ').title()}: {features.get(feat, 0):.2f}"
            for feat, contrib in sorted_contrib
            if features.get(feat, 0) < 0.2 and weights[feat] > 0.15
        ]
        
        return {
            'primary_factors': primary[:3],  # Top 3
            'mitigating_factors': mitigating[:2],  # Top 2
            'corridor_context': f"Corridor: {corridor_id}",
            'score_breakdown': {
                feat: round(contrib, 4)
                for feat, contrib in sorted_contrib
            }
        }
    
    def update_infrastructure_status(self, corridor_id: str, health: float):
        """
        Update infrastructure health status for a corridor.
        
        Args:
            corridor_id: Payment corridor
            health: Health score 0-1 (1 = fully healthy)
        """
        self.infrastructure_status[corridor_id] = max(0, min(1, health))
    
    def batch_score(
        self,
        transactions: List[Dict],
        corridor_field: str = 'corridor'
    ) -> List[Dict]:
        """
        Score multiple transactions efficiently.
        
        Args:
            transactions: List of transaction dicts with features
            corridor_field: Key containing corridor ID
            
        Returns:
            List of score results
        """
        results = []
        for txn in transactions:
            corridor = txn.get(corridor_field, 'UNKNOWN')
            
            # Extract features (assuming they're in the transaction dict)
            features = {
                'amount_deviation': txn.get('amount_deviation', 0),
                'velocity': txn.get('velocity', 0),
                'temporal_anomaly': txn.get('temporal_anomaly', 0),
                'sender_maturity': txn.get('sender_maturity', 0),
                'beneficiary_novelty': txn.get('beneficiary_novelty', 0),
            }
            
            result = self.calculate_fraud_score(features, corridor)
            result['transaction_id'] = txn.get('transaction_id')
            results.append(result)
        
        return results


def learn_corridor_multipliers(
    transactions_df,
    feature_cols: List[str],
    target_col: str = 'is_fraud',
    corridor_col: str = 'corridor'
) -> Dict[str, CorridorMultipliers]:
    """
    Learn optimal multipliers from historical data.
    
    This function analyses which features are most predictive
    for each corridor and generates appropriate multipliers.
    
    Args:
        transactions_df: DataFrame with features and fraud labels
        feature_cols: List of feature column names
        target_col: Name of fraud label column
        corridor_col: Name of corridor column
        
    Returns:
        Dict mapping corridor IDs to CorridorMultipliers
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    learned_multipliers = {}
    
    for corridor in transactions_df[corridor_col].unique():
        corridor_data = transactions_df[transactions_df[corridor_col] == corridor]
        
        if len(corridor_data) < 100 or corridor_data[target_col].sum() < 10:
            # Insufficient data - use defaults
            learned_multipliers[corridor] = CorridorMultipliers()
            continue
        
        X = corridor_data[feature_cols].values
        y = corridor_data[target_col].values
        
        # Standardise features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit logistic regression to get feature importances
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_scaled, y)
        
        # Convert coefficients to multipliers
        # Higher coefficient = more important = higher multiplier
        coefs = np.abs(model.coef_[0])
        
        # Normalise to mean of 1.0
        normalised_coefs = coefs / coefs.mean()
        
        # Cap extreme values
        capped_coefs = np.clip(normalised_coefs, 0.5, 2.0)
        
        # Create multipliers object
        multiplier_dict = dict(zip(feature_cols, capped_coefs))
        
        learned_multipliers[corridor] = CorridorMultipliers(
            amount_deviation=multiplier_dict.get('amount_deviation', 1.0),
            velocity=multiplier_dict.get('velocity', 1.0),
            temporal_anomaly=multiplier_dict.get('temporal_anomaly', 1.0),
            sender_maturity=multiplier_dict.get('sender_maturity', 1.0),
            beneficiary_novelty=multiplier_dict.get('beneficiary_novelty', 1.0),
        )
    
    return learned_multipliers


# Example usage and testing
if __name__ == '__main__':
    print("=== Dynamic Signal Weighting Demo ===\n")
    
    # Initialise calculator
    calculator = DynamicWeightCalculator()
    
    # Show weight comparison
    print("Weight comparison across corridors:\n")
    comparison = calculator.get_weight_comparison()
    
    for corridor, weights in comparison.items():
        print(f"{corridor}:")
        for feature, weight in weights.items():
            print(f"  {feature}: {weight:.3f}")
        print()
    
    # Demo scoring
    print("=== Scoring Demo ===\n")
    
    scorer = FraudScorer(weight_calculator=calculator)
    
    # Example transaction features
    sample_features = {
        'amount_deviation': 0.6,
        'velocity': 0.3,
        'temporal_anomaly': 0.1,
        'sender_maturity': 0.2,
        'beneficiary_novelty': 0.7,
    }
    
    # Score same transaction in different corridors
    for corridor in ['GBP_NGN', 'GBP_PLN', 'GBP_INR']:
        result = scorer.calculate_fraud_score(sample_features, corridor)
        print(f"Corridor: {corridor}")
        print(f"  Score: {result['score']:.3f}")
        print(f"  Decision: {result['decision']}")
        print(f"  Primary factors: {result['explanation']['primary_factors']}")
        print()
