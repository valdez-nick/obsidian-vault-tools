"""
Ranking Engine for Obsidian Vault Tools Memory Service.

Provides advanced ranking algorithms for search results with support for
multi-criteria ranking, learning-to-rank features, and personalization.
"""

import logging
import math
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


class RankingStrategy(Enum):
    """Supported ranking strategies."""
    SCORE_BASED = "score_based"
    MULTI_CRITERIA = "multi_criteria"
    LEARNING_TO_RANK = "learning_to_rank"
    PERSONALIZED = "personalized"
    TEMPORAL = "temporal"
    DIVERSITY = "diversity"


@dataclass
class RankingCriteria:
    """Criteria for multi-criteria ranking."""
    relevance_weight: float = 0.4
    recency_weight: float = 0.2
    popularity_weight: float = 0.2
    diversity_weight: float = 0.1
    personalization_weight: float = 0.1
    
    def __post_init__(self):
        """Normalize weights to sum to 1.0."""
        total = (self.relevance_weight + self.recency_weight + self.popularity_weight + 
                self.diversity_weight + self.personalization_weight)
        if total > 0:
            self.relevance_weight /= total
            self.recency_weight /= total
            self.popularity_weight /= total
            self.diversity_weight /= total
            self.personalization_weight /= total


@dataclass
class RankingFeatures:
    """Features used for ranking calculations."""
    relevance_score: float
    recency_score: float = 0.0
    popularity_score: float = 0.0
    diversity_score: float = 0.0
    personalization_score: float = 0.0
    metadata_boost: float = 0.0
    query_match_score: float = 0.0
    user_interaction_score: float = 0.0
    
    def to_vector(self) -> List[float]:
        """Convert to feature vector for ML models."""
        return [
            self.relevance_score,
            self.recency_score,
            self.popularity_score,
            self.diversity_score,
            self.personalization_score,
            self.metadata_boost,
            self.query_match_score,
            self.user_interaction_score
        ]


@dataclass
class RankedResult:
    """Result with ranking information."""
    document_id: str
    content: str
    original_score: float
    final_score: float
    rank: int
    features: RankingFeatures
    explanation: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'document_id': self.document_id,
            'content': self.content,
            'original_score': self.original_score,
            'final_score': self.final_score,
            'rank': self.rank,
            'features': {
                'relevance_score': self.features.relevance_score,
                'recency_score': self.features.recency_score,
                'popularity_score': self.features.popularity_score,
                'diversity_score': self.features.diversity_score,
                'personalization_score': self.features.personalization_score,
                'metadata_boost': self.features.metadata_boost,
                'query_match_score': self.features.query_match_score,
                'user_interaction_score': self.features.user_interaction_score
            },
            'explanation': self.explanation,
            'metadata': self.metadata
        }


class UserProfile:
    """User profile for personalized ranking."""
    
    def __init__(self, user_id: str):
        """Initialize user profile."""
        self.user_id = user_id
        self.preferences: Dict[str, float] = {}
        self.interaction_history: List[Dict] = []
        self.topic_interests: Dict[str, float] = {}
        self.document_ratings: Dict[str, float] = {}
        self.last_updated = datetime.now()
    
    def update_preference(self, key: str, value: float):
        """Update user preference."""
        self.preferences[key] = value
        self.last_updated = datetime.now()
    
    def add_interaction(self, document_id: str, interaction_type: str, score: float = 1.0):
        """Add user interaction record."""
        interaction = {
            'document_id': document_id,
            'interaction_type': interaction_type,
            'score': score,
            'timestamp': datetime.now().isoformat()
        }
        self.interaction_history.append(interaction)
        
        # Keep only recent interactions (last 1000)
        if len(self.interaction_history) > 1000:
            self.interaction_history = self.interaction_history[-1000:]
    
    def get_document_score(self, document_id: str) -> float:
        """Get personalized score for a document."""
        # Calculate based on interaction history
        score = 0.0
        for interaction in self.interaction_history:
            if interaction['document_id'] == document_id:
                score += interaction['score']
        
        # Add explicit rating if available
        if document_id in self.document_ratings:
            score += self.document_ratings[document_id] * 2
        
        return min(score, 5.0)  # Cap at 5.0


class RankingEngine:
    """
    Advanced ranking engine with multiple strategies.
    
    Features:
    - Multi-criteria ranking with configurable weights
    - Temporal ranking considering recency
    - Diversity-aware ranking to avoid redundancy
    - Personalized ranking based on user profiles
    - Learning-to-rank capabilities
    - Popularity-based boosting
    - Metadata-driven ranking adjustments
    """
    
    def __init__(self):
        """Initialize ranking engine."""
        self.default_criteria = RankingCriteria()
        self.user_profiles: Dict[str, UserProfile] = {}
        self.document_stats: Dict[str, Dict] = {}
        self.popularity_stats: Dict[str, float] = {}
        
        # Learning-to-rank model (placeholder for ML integration)
        self.ltr_model = None
        self.ltr_features = []
        
        logger.info("Initialized RankingEngine with multiple strategies")
    
    def rank_results(
        self,
        results: List[Dict[str, Any]],
        query: str,
        strategy: RankingStrategy = RankingStrategy.MULTI_CRITERIA,
        criteria: Optional[RankingCriteria] = None,
        user_id: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> List[RankedResult]:
        """
        Rank search results using specified strategy.
        
        Args:
            results: List of search results with scores
            query: Original search query
            strategy: Ranking strategy to use
            criteria: Custom ranking criteria
            user_id: User ID for personalized ranking
            context: Additional context for ranking
            
        Returns:
            List of ranked results
        """
        try:
            if not results:
                return []
            
            # Use default criteria if none provided
            ranking_criteria = criteria or self.default_criteria
            
            # Calculate features for each result
            ranked_results = []
            
            for i, result in enumerate(results):
                features = self._calculate_features(
                    result=result,
                    query=query,
                    user_id=user_id,
                    context=context,
                    position=i
                )
                
                # Calculate final score based on strategy
                final_score = self._calculate_final_score(
                    original_score=result.get('score', 0.0),
                    features=features,
                    strategy=strategy,
                    criteria=ranking_criteria
                )
                
                # Generate explanation
                explanation = self._generate_ranking_explanation(
                    strategy=strategy,
                    features=features,
                    criteria=ranking_criteria
                )
                
                ranked_result = RankedResult(
                    document_id=result.get('document_id', ''),
                    content=result.get('content', ''),
                    original_score=result.get('score', 0.0),
                    final_score=final_score,
                    rank=0,  # Will be set after sorting
                    features=features,
                    explanation=explanation,
                    metadata=result.get('metadata', {})
                )
                
                ranked_results.append(ranked_result)
            
            # Sort by final score
            ranked_results.sort(key=lambda x: x.final_score, reverse=True)
            
            # Apply diversity filtering if needed
            if strategy == RankingStrategy.DIVERSITY:
                ranked_results = self._apply_diversity_filtering(ranked_results, query)
            
            # Set final ranks
            for i, result in enumerate(ranked_results):
                result.rank = i + 1
            
            logger.debug(f"Ranked {len(ranked_results)} results using {strategy.value} strategy")
            
            return ranked_results
            
        except Exception as e:
            logger.error(f"Ranking failed: {e}")
            raise
    
    def _calculate_features(
        self,
        result: Dict[str, Any],
        query: str,
        user_id: Optional[str],
        context: Optional[Dict],
        position: int
    ) -> RankingFeatures:
        """Calculate ranking features for a result."""
        features = RankingFeatures(
            relevance_score=result.get('score', 0.0)
        )
        
        doc_id = result.get('document_id', '')
        metadata = result.get('metadata', {})
        content = result.get('content', '')
        
        # Recency score
        features.recency_score = self._calculate_recency_score(metadata)
        
        # Popularity score
        features.popularity_score = self._calculate_popularity_score(doc_id)
        
        # Diversity score (position-based for now)
        features.diversity_score = self._calculate_diversity_score(result, position)
        
        # Personalization score
        if user_id:
            features.personalization_score = self._calculate_personalization_score(
                doc_id, user_id, metadata
            )
        
        # Metadata boost
        features.metadata_boost = self._calculate_metadata_boost(metadata, query)
        
        # Query match score
        features.query_match_score = self._calculate_query_match_score(content, query)
        
        # User interaction score
        if user_id:
            features.user_interaction_score = self._calculate_user_interaction_score(
                doc_id, user_id
            )
        
        return features
    
    def _calculate_recency_score(self, metadata: Dict) -> float:
        """Calculate recency score based on document timestamp."""
        try:
            # Look for timestamp in metadata
            timestamp_field = metadata.get('timestamp') or metadata.get('created_time') or metadata.get('modified_time')
            
            if not timestamp_field:
                return 0.5  # Neutral score for unknown age
            
            # Parse timestamp
            if isinstance(timestamp_field, str):
                try:
                    doc_time = datetime.fromisoformat(timestamp_field.replace('Z', '+00:00'))
                except:
                    return 0.5
            elif isinstance(timestamp_field, (int, float)):
                doc_time = datetime.fromtimestamp(timestamp_field)
            else:
                return 0.5
            
            # Calculate age in days
            age_days = (datetime.now() - doc_time).days
            
            # Convert to score (more recent = higher score)
            # Documents from last week get full score, linearly decreasing
            if age_days <= 7:
                return 1.0
            elif age_days <= 30:
                return 1.0 - (age_days - 7) / 23 * 0.5  # 0.5-1.0 range
            elif age_days <= 365:
                return 0.5 - (age_days - 30) / 335 * 0.4  # 0.1-0.5 range
            else:
                return 0.1  # Very old documents
                
        except Exception:
            return 0.5  # Default neutral score
    
    def _calculate_popularity_score(self, doc_id: str) -> float:
        """Calculate popularity score based on access statistics."""
        return self.popularity_stats.get(doc_id, 0.0)
    
    def _calculate_diversity_score(self, result: Dict, position: int) -> float:
        """Calculate diversity score to promote result variety."""
        # Simple position-based diversity (could be enhanced with content analysis)
        # Later results get slight boost to promote diversity
        base_score = 0.5
        position_boost = min(position * 0.05, 0.3)  # Max 0.3 boost
        
        # Could add topic clustering analysis here
        return base_score + position_boost
    
    def _calculate_personalization_score(
        self, 
        doc_id: str, 
        user_id: str, 
        metadata: Dict
    ) -> float:
        """Calculate personalized score based on user profile."""
        if user_id not in self.user_profiles:
            return 0.0
        
        profile = self.user_profiles[user_id]
        
        # Base score from document interactions
        base_score = profile.get_document_score(doc_id) / 5.0  # Normalize to 0-1
        
        # Add topic interest alignment
        topic_score = 0.0
        doc_topics = metadata.get('tags', [])
        if isinstance(doc_topics, list) and profile.topic_interests:
            for topic in doc_topics:
                if topic in profile.topic_interests:
                    topic_score += profile.topic_interests[topic]
            
            if doc_topics:
                topic_score /= len(doc_topics)  # Average topic interest
        
        # Combine scores
        final_score = base_score * 0.7 + topic_score * 0.3
        return min(final_score, 1.0)
    
    def _calculate_metadata_boost(self, metadata: Dict, query: str) -> float:
        """Calculate boost based on metadata relevance."""
        boost = 0.0
        query_lower = query.lower()
        
        # Title/filename boost
        title = metadata.get('title', '') or metadata.get('filename', '')
        if title and query_lower in title.lower():
            boost += 0.3
        
        # Tag boost
        tags = metadata.get('tags', [])
        if isinstance(tags, list):
            for tag in tags:
                if query_lower in tag.lower():
                    boost += 0.1
                    break
        
        # Type boost (prefer certain content types)
        content_type = metadata.get('type', '')
        if content_type in ['note', 'document', 'article']:
            boost += 0.1
        
        return min(boost, 0.5)  # Cap at 0.5
    
    def _calculate_query_match_score(self, content: str, query: str) -> float:
        """Calculate query match score based on content analysis."""
        if not content or not query:
            return 0.0
        
        content_lower = content.lower()
        query_lower = query.lower()
        query_words = query_lower.split()
        
        # Exact phrase match
        if query_lower in content_lower:
            return 1.0
        
        # Word overlap
        content_words = set(content_lower.split())
        query_word_set = set(query_words)
        overlap = len(content_words & query_word_set)
        
        if query_words:
            return overlap / len(query_words)
        
        return 0.0
    
    def _calculate_user_interaction_score(self, doc_id: str, user_id: str) -> float:
        """Calculate score based on user interactions with document."""
        if user_id not in self.user_profiles:
            return 0.0
        
        profile = self.user_profiles[user_id]
        
        # Count recent interactions
        recent_interactions = 0
        for interaction in profile.interaction_history[-100:]:  # Last 100 interactions
            if interaction['document_id'] == doc_id:
                recent_interactions += interaction['score']
        
        return min(recent_interactions / 10.0, 1.0)  # Normalize and cap
    
    def _calculate_final_score(
        self,
        original_score: float,
        features: RankingFeatures,
        strategy: RankingStrategy,
        criteria: RankingCriteria
    ) -> float:
        """Calculate final ranking score based on strategy."""
        if strategy == RankingStrategy.SCORE_BASED:
            return original_score
        
        elif strategy == RankingStrategy.MULTI_CRITERIA:
            # Weighted combination of all features
            final_score = (
                criteria.relevance_weight * features.relevance_score +
                criteria.recency_weight * features.recency_score +
                criteria.popularity_weight * features.popularity_score +
                criteria.diversity_weight * features.diversity_score +
                criteria.personalization_weight * features.personalization_score
            )
            
            # Add boosts
            final_score += features.metadata_boost + features.query_match_score * 0.1
            
            return min(final_score, 1.0)
        
        elif strategy == RankingStrategy.LEARNING_TO_RANK:
            if self.ltr_model:
                # Use ML model to predict score
                feature_vector = features.to_vector()
                return self.ltr_model.predict([feature_vector])[0]
            else:
                # Fallback to multi-criteria
                return self._calculate_final_score(
                    original_score, features, RankingStrategy.MULTI_CRITERIA, criteria
                )
        
        elif strategy == RankingStrategy.PERSONALIZED:
            # Heavy emphasis on personalization
            base_score = original_score * 0.5
            personal_score = (
                features.personalization_score * 0.3 +
                features.user_interaction_score * 0.2
            )
            return base_score + personal_score
        
        elif strategy == RankingStrategy.TEMPORAL:
            # Heavy emphasis on recency
            return original_score * 0.6 + features.recency_score * 0.4
        
        elif strategy == RankingStrategy.DIVERSITY:
            # Promote diversity while maintaining relevance
            return original_score * 0.7 + features.diversity_score * 0.3
        
        else:
            return original_score
    
    def _apply_diversity_filtering(
        self, 
        results: List[RankedResult], 
        query: str,
        similarity_threshold: float = 0.8
    ) -> List[RankedResult]:
        """Apply diversity filtering to reduce redundant results."""
        if len(results) <= 3:
            return results  # Too few results to filter
        
        filtered_results = [results[0]]  # Always keep top result
        
        for candidate in results[1:]:
            # Check similarity with already selected results
            is_diverse = True
            
            for selected in filtered_results:
                # Simple content similarity check (could use embeddings)
                similarity = self._calculate_content_similarity(
                    candidate.content, selected.content
                )
                
                if similarity > similarity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                filtered_results.append(candidate)
            
            # Stop if we have enough diverse results
            if len(filtered_results) >= min(10, len(results)):
                break
        
        return filtered_results
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate simple content similarity."""
        if not content1 or not content2:
            return 0.0
        
        # Simple word overlap similarity
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _generate_ranking_explanation(
        self,
        strategy: RankingStrategy,
        features: RankingFeatures,
        criteria: RankingCriteria
    ) -> str:
        """Generate human-readable ranking explanation."""
        explanations = []
        
        # Strategy explanation
        if strategy == RankingStrategy.MULTI_CRITERIA:
            explanations.append("Multi-criteria ranking")
            
            # Highlight dominant factors
            scores = [
                ("relevance", features.relevance_score, criteria.relevance_weight),
                ("recency", features.recency_score, criteria.recency_weight),
                ("popularity", features.popularity_score, criteria.popularity_weight),
                ("personalization", features.personalization_score, criteria.personalization_weight)
            ]
            
            # Find top contributing factors
            weighted_scores = [(name, score * weight) for name, score, weight in scores if score > 0]
            weighted_scores.sort(key=lambda x: x[1], reverse=True)
            
            if weighted_scores:
                top_factors = [name for name, _ in weighted_scores[:2]]
                explanations.append(f"Top factors: {', '.join(top_factors)}")
        
        elif strategy == RankingStrategy.PERSONALIZED:
            explanations.append("Personalized ranking")
            if features.personalization_score > 0.5:
                explanations.append("High personal relevance")
        
        elif strategy == RankingStrategy.TEMPORAL:
            explanations.append("Recent content prioritized")
        
        elif strategy == RankingStrategy.DIVERSITY:
            explanations.append("Diversity-enhanced ranking")
        
        # Add boost explanations
        if features.metadata_boost > 0.2:
            explanations.append("Metadata match boost")
        
        if features.query_match_score > 0.7:
            explanations.append("Strong query match")
        
        return "; ".join(explanations) if explanations else "Standard ranking"
    
    def update_document_stats(self, doc_id: str, interaction_type: str, user_id: Optional[str] = None):
        """Update document statistics for popularity calculation."""
        if doc_id not in self.document_stats:
            self.document_stats[doc_id] = {
                'views': 0,
                'clicks': 0,
                'ratings': [],
                'last_accessed': datetime.now()
            }
        
        stats = self.document_stats[doc_id]
        
        if interaction_type == 'view':
            stats['views'] += 1
        elif interaction_type == 'click':
            stats['clicks'] += 1
        
        stats['last_accessed'] = datetime.now()
        
        # Update popularity score (simple view-based for now)
        total_interactions = stats['views'] + stats['clicks'] * 2
        self.popularity_stats[doc_id] = min(total_interactions / 100.0, 1.0)
        
        # Update user profile if provided
        if user_id:
            self.update_user_profile(user_id, doc_id, interaction_type)
    
    def update_user_profile(self, user_id: str, doc_id: str, interaction_type: str, score: float = 1.0):
        """Update user profile with interaction."""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id)
        
        profile = self.user_profiles[user_id]
        profile.add_interaction(doc_id, interaction_type, score)
    
    def set_user_preference(self, user_id: str, preference_key: str, value: float):
        """Set user preference for personalized ranking."""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id)
        
        self.user_profiles[user_id].update_preference(preference_key, value)
    
    def train_ltr_model(self, training_data: List[Dict], features: List[str]):
        """
        Train learning-to-rank model (placeholder for ML integration).
        
        Args:
            training_data: Training examples with features and relevance labels
            features: List of feature names to use
        """
        logger.info("LTR model training not implemented - using multi-criteria fallback")
        # Placeholder for future ML integration
        self.ltr_features = features
    
    def get_ranking_stats(self) -> Dict[str, Any]:
        """Get ranking engine statistics."""
        return {
            'total_documents': len(self.document_stats),
            'total_users': len(self.user_profiles),
            'popular_documents': sorted(
                self.popularity_stats.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10],
            'default_criteria': {
                'relevance_weight': self.default_criteria.relevance_weight,
                'recency_weight': self.default_criteria.recency_weight,
                'popularity_weight': self.default_criteria.popularity_weight,
                'diversity_weight': self.default_criteria.diversity_weight,
                'personalization_weight': self.default_criteria.personalization_weight
            }
        }
    
    def cleanup(self):
        """Clean up ranking engine resources."""
        self.user_profiles.clear()
        self.document_stats.clear()
        self.popularity_stats.clear()
        logger.info("RankingEngine cleanup completed")