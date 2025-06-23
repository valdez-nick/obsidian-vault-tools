"""
Similarity Calculator for Obsidian Vault Tools Memory Service.

Provides various similarity metrics for vector embeddings and text content
with support for custom distance functions and similarity explanations.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import math

logger = logging.getLogger(__name__)


class SimilarityMetric(Enum):
    """Supported similarity metrics."""
    COSINE = "cosine"
    DOT_PRODUCT = "dot_product" 
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    JACCARD = "jaccard"
    HAMMING = "hamming"
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    JENSEN_SHANNON = "jensen_shannon"


@dataclass
class SimilarityResult:
    """Result of similarity calculation."""
    metric: SimilarityMetric
    score: float
    distance: float
    explanation: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'metric': self.metric.value,
            'score': self.score,
            'distance': self.distance,
            'explanation': self.explanation,
            'metadata': self.metadata
        }


class SimilarityCalculator:
    """
    Advanced similarity calculator with multiple metrics.
    
    Features:
    - Multiple similarity metrics (cosine, dot product, euclidean, etc.)
    - Batch similarity calculations
    - Custom distance functions
    - Similarity explanations
    - Performance optimization
    - Statistical analysis
    """
    
    def __init__(self):
        """Initialize similarity calculator."""
        self._metric_functions = {
            SimilarityMetric.COSINE: self._cosine_similarity,
            SimilarityMetric.DOT_PRODUCT: self._dot_product_similarity,
            SimilarityMetric.EUCLIDEAN: self._euclidean_similarity,
            SimilarityMetric.MANHATTAN: self._manhattan_similarity,
            SimilarityMetric.JACCARD: self._jaccard_similarity,
            SimilarityMetric.HAMMING: self._hamming_similarity,
            SimilarityMetric.PEARSON: self._pearson_similarity,
            SimilarityMetric.SPEARMAN: self._spearman_similarity,
            SimilarityMetric.JENSEN_SHANNON: self._jensen_shannon_similarity
        }
        
        logger.info("Initialized SimilarityCalculator with multiple metrics")
    
    def calculate_similarity(
        self,
        vector1: Union[np.ndarray, List[float]],
        vector2: Union[np.ndarray, List[float]],
        metric: SimilarityMetric = SimilarityMetric.COSINE,
        normalize: bool = True
    ) -> SimilarityResult:
        """
        Calculate similarity between two vectors.
        
        Args:
            vector1: First vector
            vector2: Second vector
            metric: Similarity metric to use
            normalize: Whether to normalize vectors
            
        Returns:
            SimilarityResult with score and explanation
        """
        try:
            # Convert to numpy arrays
            v1 = np.array(vector1, dtype=np.float32)
            v2 = np.array(vector2, dtype=np.float32)
            
            # Validate dimensions
            if v1.shape != v2.shape:
                raise ValueError(f"Vector dimensions don't match: {v1.shape} vs {v2.shape}")
            
            # Normalize if requested
            if normalize and metric in [SimilarityMetric.COSINE, SimilarityMetric.DOT_PRODUCT]:
                v1 = self._normalize_vector(v1)
                v2 = self._normalize_vector(v2)
            
            # Calculate similarity using appropriate function
            similarity_func = self._metric_functions.get(metric)
            if not similarity_func:
                raise ValueError(f"Unsupported similarity metric: {metric}")
            
            score, distance, metadata = similarity_func(v1, v2)
            
            # Generate explanation
            explanation = self._generate_similarity_explanation(metric, score, metadata)
            
            return SimilarityResult(
                metric=metric,
                score=score,
                distance=distance,
                explanation=explanation,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            raise
    
    def batch_similarity(
        self,
        query_vector: Union[np.ndarray, List[float]],
        candidate_vectors: Union[np.ndarray, List[List[float]]],
        metric: SimilarityMetric = SimilarityMetric.COSINE,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[Tuple[int, SimilarityResult]]:
        """
        Calculate similarity between query and multiple candidates.
        
        Args:
            query_vector: Query vector
            candidate_vectors: List/array of candidate vectors
            metric: Similarity metric to use
            top_k: Return top-k most similar
            threshold: Minimum similarity threshold
            
        Returns:
            List of (index, SimilarityResult) tuples
        """
        try:
            query = np.array(query_vector, dtype=np.float32)
            candidates = np.array(candidate_vectors, dtype=np.float32)
            
            if len(candidates.shape) != 2:
                raise ValueError("Candidate vectors must be 2D array")
            
            if query.shape[0] != candidates.shape[1]:
                raise ValueError(f"Dimension mismatch: query {query.shape[0]} vs candidates {candidates.shape[1]}")
            
            results = []
            
            for i, candidate in enumerate(candidates):
                try:
                    sim_result = self.calculate_similarity(query, candidate, metric)
                    
                    # Apply threshold if specified
                    if threshold is None or sim_result.score >= threshold:
                        results.append((i, sim_result))
                        
                except Exception as e:
                    logger.warning(f"Failed to calculate similarity for candidate {i}: {e}")
                    continue
            
            # Sort by similarity score (descending)
            results.sort(key=lambda x: x[1].score, reverse=True)
            
            # Apply top-k limit
            if top_k is not None:
                results = results[:top_k]
            
            return results
            
        except Exception as e:
            logger.error(f"Batch similarity calculation failed: {e}")
            raise
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
    
    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> Tuple[float, float, Dict]:
        """Calculate cosine similarity."""
        # Normalize vectors
        v1_norm = self._normalize_vector(v1)
        v2_norm = self._normalize_vector(v2)
        
        # Calculate dot product of normalized vectors
        dot_product = np.dot(v1_norm, v2_norm)
        
        # Clamp to [-1, 1] to handle numerical errors
        similarity = np.clip(dot_product, -1.0, 1.0)
        distance = 1.0 - similarity
        
        metadata = {
            'dot_product': float(dot_product),
            'v1_norm': float(np.linalg.norm(v1)),
            'v2_norm': float(np.linalg.norm(v2))
        }
        
        return float(similarity), float(distance), metadata
    
    def _dot_product_similarity(self, v1: np.ndarray, v2: np.ndarray) -> Tuple[float, float, Dict]:
        """Calculate dot product similarity."""
        dot_product = np.dot(v1, v2)
        
        # For similarity, we return the dot product directly
        # Distance is the negative dot product (higher dot product = lower distance)
        similarity = float(dot_product)
        distance = -similarity
        
        metadata = {
            'raw_dot_product': similarity,
            'v1_magnitude': float(np.linalg.norm(v1)),
            'v2_magnitude': float(np.linalg.norm(v2))
        }
        
        return similarity, distance, metadata
    
    def _euclidean_similarity(self, v1: np.ndarray, v2: np.ndarray) -> Tuple[float, float, Dict]:
        """Calculate Euclidean distance-based similarity."""
        distance = float(np.linalg.norm(v1 - v2))
        
        # Convert distance to similarity (closer = more similar)
        # Using negative exponential to map distance to [0, 1]
        similarity = math.exp(-distance)
        
        metadata = {
            'euclidean_distance': distance,
            'dimension': len(v1)
        }
        
        return similarity, distance, metadata
    
    def _manhattan_similarity(self, v1: np.ndarray, v2: np.ndarray) -> Tuple[float, float, Dict]:
        """Calculate Manhattan distance-based similarity."""
        distance = float(np.sum(np.abs(v1 - v2)))
        
        # Convert to similarity
        similarity = math.exp(-distance / len(v1))  # Normalize by dimension
        
        metadata = {
            'manhattan_distance': distance,
            'normalized_distance': distance / len(v1)
        }
        
        return similarity, distance, metadata
    
    def _jaccard_similarity(self, v1: np.ndarray, v2: np.ndarray) -> Tuple[float, float, Dict]:
        """Calculate Jaccard similarity (for binary/sparse vectors)."""
        # Binarize vectors (treat non-zero as 1)
        b1 = (v1 != 0).astype(int)
        b2 = (v2 != 0).astype(int)
        
        intersection = np.sum(b1 * b2)
        union = np.sum(np.logical_or(b1, b2).astype(int))
        
        if union == 0:
            similarity = 1.0  # Both vectors are zero
        else:
            similarity = float(intersection / union)
        
        distance = 1.0 - similarity
        
        metadata = {
            'intersection': int(intersection),
            'union': int(union),
            'v1_nonzero': int(np.sum(b1)),
            'v2_nonzero': int(np.sum(b2))
        }
        
        return similarity, distance, metadata
    
    def _hamming_similarity(self, v1: np.ndarray, v2: np.ndarray) -> Tuple[float, float, Dict]:
        """Calculate Hamming distance-based similarity (for binary vectors)."""
        # Binarize vectors
        b1 = (v1 != 0).astype(int)
        b2 = (v2 != 0).astype(int)
        
        # Count differing positions
        distance = float(np.sum(b1 != b2))
        
        # Convert to similarity (normalize by vector length)
        similarity = 1.0 - (distance / len(v1))
        
        metadata = {
            'hamming_distance': distance,
            'total_positions': len(v1),
            'matching_positions': len(v1) - distance
        }
        
        return similarity, distance, metadata
    
    def _pearson_similarity(self, v1: np.ndarray, v2: np.ndarray) -> Tuple[float, float, Dict]:
        """Calculate Pearson correlation coefficient."""
        try:
            correlation_matrix = np.corrcoef(v1, v2)
            correlation = correlation_matrix[0, 1]
            
            # Handle NaN values (e.g., when vectors have zero variance)
            if np.isnan(correlation):
                correlation = 0.0
            
            similarity = float(correlation)
            distance = 1.0 - abs(similarity)  # Use absolute value for distance
            
            metadata = {
                'correlation': similarity,
                'v1_mean': float(np.mean(v1)),
                'v2_mean': float(np.mean(v2)),
                'v1_std': float(np.std(v1)),
                'v2_std': float(np.std(v2))
            }
            
            return similarity, distance, metadata
            
        except Exception:
            # Fallback for edge cases
            return 0.0, 1.0, {'error': 'Pearson calculation failed'}
    
    def _spearman_similarity(self, v1: np.ndarray, v2: np.ndarray) -> Tuple[float, float, Dict]:
        """Calculate Spearman rank correlation."""
        try:
            from scipy.stats import spearmanr
            
            correlation, p_value = spearmanr(v1, v2)
            
            if np.isnan(correlation):
                correlation = 0.0
            
            similarity = float(correlation)
            distance = 1.0 - abs(similarity)
            
            metadata = {
                'spearman_correlation': similarity,
                'p_value': float(p_value) if not np.isnan(p_value) else None
            }
            
            return similarity, distance, metadata
            
        except ImportError:
            logger.warning("scipy not available for Spearman correlation, falling back to Pearson")
            return self._pearson_similarity(v1, v2)
        except Exception:
            return 0.0, 1.0, {'error': 'Spearman calculation failed'}
    
    def _jensen_shannon_similarity(self, v1: np.ndarray, v2: np.ndarray) -> Tuple[float, float, Dict]:
        """Calculate Jensen-Shannon divergence-based similarity."""
        try:
            # Normalize vectors to probability distributions
            p1 = v1 / np.sum(v1) if np.sum(v1) > 0 else np.ones_like(v1) / len(v1)
            p2 = v2 / np.sum(v2) if np.sum(v2) > 0 else np.ones_like(v2) / len(v2)
            
            # Calculate Jensen-Shannon divergence
            m = 0.5 * (p1 + p2)
            
            # KL divergences
            kl1 = np.sum(p1 * np.log(p1 / m + 1e-10))
            kl2 = np.sum(p2 * np.log(p2 / m + 1e-10))
            
            js_divergence = 0.5 * kl1 + 0.5 * kl2
            
            # Convert to similarity (0 divergence = 1 similarity)
            similarity = 1.0 - min(js_divergence, 1.0)
            distance = float(js_divergence)
            
            metadata = {
                'js_divergence': distance,
                'kl_divergence_1': float(kl1),
                'kl_divergence_2': float(kl2)
            }
            
            return similarity, distance, metadata
            
        except Exception:
            return 0.0, 1.0, {'error': 'Jensen-Shannon calculation failed'}
    
    def _generate_similarity_explanation(
        self,
        metric: SimilarityMetric,
        score: float,
        metadata: Dict
    ) -> str:
        """Generate human-readable explanation of similarity score."""
        explanations = []
        
        # Score interpretation
        if score >= 0.9:
            explanations.append("Very high similarity")
        elif score >= 0.8:
            explanations.append("High similarity")
        elif score >= 0.7:
            explanations.append("Good similarity")
        elif score >= 0.6:
            explanations.append("Moderate similarity")
        elif score >= 0.4:
            explanations.append("Low similarity")
        else:
            explanations.append("Very low similarity")
        
        # Metric-specific explanations
        if metric == SimilarityMetric.COSINE:
            explanations.append(f"Cosine similarity: {score:.3f}")
            if 'dot_product' in metadata:
                explanations.append(f"Dot product: {metadata['dot_product']:.3f}")
        
        elif metric == SimilarityMetric.EUCLIDEAN:
            if 'euclidean_distance' in metadata:
                explanations.append(f"Euclidean distance: {metadata['euclidean_distance']:.3f}")
        
        elif metric == SimilarityMetric.JACCARD:
            if 'intersection' in metadata and 'union' in metadata:
                explanations.append(f"Jaccard: {metadata['intersection']}/{metadata['union']} overlap")
        
        elif metric == SimilarityMetric.PEARSON:
            if score > 0:
                explanations.append("Positive correlation")
            elif score < 0:
                explanations.append("Negative correlation")
            else:
                explanations.append("No correlation")
        
        return "; ".join(explanations)
    
    def compare_metrics(
        self,
        vector1: Union[np.ndarray, List[float]],
        vector2: Union[np.ndarray, List[float]],
        metrics: Optional[List[SimilarityMetric]] = None
    ) -> Dict[SimilarityMetric, SimilarityResult]:
        """
        Compare similarity using multiple metrics.
        
        Args:
            vector1: First vector
            vector2: Second vector
            metrics: List of metrics to compare (defaults to all)
            
        Returns:
            Dictionary mapping metrics to results
        """
        if metrics is None:
            metrics = list(SimilarityMetric)
        
        results = {}
        
        for metric in metrics:
            try:
                result = self.calculate_similarity(vector1, vector2, metric)
                results[metric] = result
            except Exception as e:
                logger.warning(f"Failed to calculate {metric.value} similarity: {e}")
                continue
        
        return results
    
    def find_optimal_threshold(
        self,
        query_vector: Union[np.ndarray, List[float]],
        positive_examples: List[Union[np.ndarray, List[float]]],
        negative_examples: List[Union[np.ndarray, List[float]]],
        metric: SimilarityMetric = SimilarityMetric.COSINE
    ) -> Dict[str, Any]:
        """
        Find optimal similarity threshold using positive and negative examples.
        
        Args:
            query_vector: Reference query vector
            positive_examples: Vectors that should be similar
            negative_examples: Vectors that should be dissimilar
            metric: Similarity metric to use
            
        Returns:
            Dictionary with optimal threshold and analysis
        """
        try:
            # Calculate similarities for positive examples
            pos_scores = []
            for example in positive_examples:
                result = self.calculate_similarity(query_vector, example, metric)
                pos_scores.append(result.score)
            
            # Calculate similarities for negative examples
            neg_scores = []
            for example in negative_examples:
                result = self.calculate_similarity(query_vector, example, metric)
                neg_scores.append(result.score)
            
            if not pos_scores or not neg_scores:
                raise ValueError("Need both positive and negative examples")
            
            # Find threshold that maximizes separation
            all_scores = sorted(pos_scores + neg_scores)
            best_threshold = 0.5
            best_accuracy = 0.0
            
            for threshold in all_scores:
                # Count correct classifications
                correct = 0
                total = len(pos_scores) + len(neg_scores)
                
                # Positive examples should be above threshold
                correct += sum(1 for score in pos_scores if score >= threshold)
                # Negative examples should be below threshold
                correct += sum(1 for score in neg_scores if score < threshold)
                
                accuracy = correct / total
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_threshold = threshold
            
            return {
                'optimal_threshold': best_threshold,
                'accuracy': best_accuracy,
                'positive_scores': {
                    'mean': np.mean(pos_scores),
                    'std': np.std(pos_scores),
                    'min': min(pos_scores),
                    'max': max(pos_scores)
                },
                'negative_scores': {
                    'mean': np.mean(neg_scores),
                    'std': np.std(neg_scores),
                    'min': min(neg_scores),
                    'max': max(neg_scores)
                },
                'separation': np.mean(pos_scores) - np.mean(neg_scores)
            }
            
        except Exception as e:
            logger.error(f"Threshold optimization failed: {e}")
            raise
    
    def get_supported_metrics(self) -> List[SimilarityMetric]:
        """Get list of supported similarity metrics."""
        return list(SimilarityMetric)
    
    def benchmark_metrics(
        self,
        test_vectors: List[Tuple[np.ndarray, np.ndarray]],
        iterations: int = 100
    ) -> Dict[SimilarityMetric, Dict[str, float]]:
        """
        Benchmark performance of different similarity metrics.
        
        Args:
            test_vectors: List of vector pairs for testing
            iterations: Number of iterations per metric
            
        Returns:
            Dictionary with performance statistics per metric
        """
        import time
        
        results = {}
        
        for metric in SimilarityMetric:
            times = []
            errors = 0
            
            for _ in range(iterations):
                for v1, v2 in test_vectors:
                    try:
                        start_time = time.time()
                        self.calculate_similarity(v1, v2, metric)
                        elapsed = time.time() - start_time
                        times.append(elapsed)
                    except Exception:
                        errors += 1
            
            if times:
                results[metric] = {
                    'avg_time_ms': np.mean(times) * 1000,
                    'std_time_ms': np.std(times) * 1000,
                    'min_time_ms': min(times) * 1000,
                    'max_time_ms': max(times) * 1000,
                    'total_calculations': len(times),
                    'errors': errors,
                    'success_rate': len(times) / (len(times) + errors)
                }
        
        return results