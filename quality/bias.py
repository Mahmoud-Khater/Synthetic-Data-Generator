"""
Bias detection for synthetic reviews.
Identifies sentiment skew and unrealistic patterns.
"""
from typing import List, Dict
import numpy as np
from collections import Counter
import re


class BiasAnalyzer:
    """Analyzes bias in synthetic reviews."""
    
    def __init__(self):
        """Initialize the bias analyzer."""
        pass
    
    def analyze_rating_distribution(self, ratings: List[int], expected_dist: Dict[int, float]) -> Dict[str, any]:
        """
        Compare actual rating distribution to expected distribution.
        
        Args:
            ratings: List of ratings from generated reviews
            expected_dist: Expected distribution from config
            
        Returns:
            Dictionary with distribution analysis
        """
        # Calculate actual distribution
        rating_counts = Counter(ratings)
        total = len(ratings)
        actual_dist = {r: rating_counts.get(r, 0) / total for r in range(1, 6)}
        
        # Calculate deviation from expected
        deviations = {}
        for rating in range(1, 6):
            expected = expected_dist.get(rating, 0)
            actual = actual_dist.get(rating, 0)
            deviation = abs(actual - expected)
            deviations[rating] = round(deviation, 4)
        
        # Calculate chi-square-like statistic
        total_deviation = sum(deviations.values())
        
        return {
            'actual_distribution': {k: round(v, 4) for k, v in actual_dist.items()},
            'expected_distribution': {k: round(v, 4) for k, v in expected_dist.items()},
            'deviations': deviations,
            'total_deviation': round(total_deviation, 4),
            'is_biased': total_deviation > 0.15  # Threshold for significant bias
        }
    
    def analyze_sentiment_consistency(self, reviews: List[Dict]) -> Dict[str, any]:
        """
        Check if sentiment matches ratings.
        
        Args:
            reviews: List of review dictionaries with 'rating' and 'text'
            
        Returns:
            Dictionary with sentiment consistency metrics
        """
        # Simple sentiment indicators
        positive_words = {
            'great', 'excellent', 'amazing', 'perfect', 'love', 'best',
            'fantastic', 'wonderful', 'awesome', 'comfortable', 'highly'
        }
        negative_words = {
            'bad', 'terrible', 'worst', 'hate', 'disappointed', 'poor',
            'awful', 'horrible', 'waste', 'uncomfortable', 'cheap'
        }
        
        inconsistencies = []
        
        for review in reviews:
            text = review['text'].lower()
            rating = review['rating']
            
            # Count sentiment words
            pos_count = sum(1 for word in positive_words if word in text)
            neg_count = sum(1 for word in negative_words if word in text)
            
            # Check for inconsistencies
            if rating >= 4 and neg_count > pos_count:
                inconsistencies.append({
                    'rating': rating,
                    'issue': 'High rating but negative sentiment',
                    'pos_words': pos_count,
                    'neg_words': neg_count
                })
            elif rating <= 2 and pos_count > neg_count:
                inconsistencies.append({
                    'rating': rating,
                    'issue': 'Low rating but positive sentiment',
                    'pos_words': pos_count,
                    'neg_words': neg_count
                })
        
        return {
            'total_reviews': len(reviews),
            'inconsistencies_found': len(inconsistencies),
            'inconsistency_rate': round(len(inconsistencies) / len(reviews), 4) if reviews else 0,
            'is_consistent': len(inconsistencies) / len(reviews) < 0.1 if reviews else True,
            'examples': inconsistencies[:5]  # Show first 5 examples
        }
    
    def detect_repetitive_patterns(self, reviews: List[str]) -> Dict[str, any]:
        """
        Detect repetitive phrases or patterns.
        
        Args:
            reviews: List of review texts
            
        Returns:
            Dictionary with pattern detection results
        """
        # Extract common phrases (3-5 words)
        phrase_counter = Counter()
        
        for review in reviews:
            words = review.lower().split()
            # Extract 3-grams and 4-grams
            for n in [3, 4]:
                for i in range(len(words) - n + 1):
                    phrase = ' '.join(words[i:i+n])
                    # Filter out very common phrases
                    if len(phrase) > 10:  # Minimum phrase length
                        phrase_counter[phrase] += 1
        
        # Find phrases that appear too frequently
        total_reviews = len(reviews)
        repetitive_phrases = []
        
        for phrase, count in phrase_counter.most_common(20):
            frequency = count / total_reviews
            if frequency > 0.15:  # Appears in >15% of reviews
                repetitive_phrases.append({
                    'phrase': phrase,
                    'count': count,
                    'frequency': round(frequency, 4)
                })
        
        # Check for identical reviews
        unique_reviews = len(set(reviews))
        duplicate_rate = 1 - (unique_reviews / total_reviews) if total_reviews > 0 else 0
        
        return {
            'total_reviews': total_reviews,
            'unique_reviews': unique_reviews,
            'duplicate_rate': round(duplicate_rate, 4),
            'repetitive_phrases': repetitive_phrases,
            'has_repetition_issues': len(repetitive_phrases) > 5 or duplicate_rate > 0.05
        }
    
    def analyze_length_distribution(self, reviews: List[str]) -> Dict[str, any]:
        """
        Analyze review length distribution for unrealistic uniformity.
        
        Args:
            reviews: List of review texts
            
        Returns:
            Dictionary with length distribution metrics
        """
        lengths = [len(review.split()) for review in reviews]
        
        return {
            'avg_length': round(np.mean(lengths), 2),
            'std_length': round(np.std(lengths), 2),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'is_too_uniform': np.std(lengths) < 10  # Very low variance is suspicious
        }
    
    def analyze(self, reviews: List[Dict], expected_rating_dist: Dict[int, float]) -> Dict[str, any]:
        """
        Perform complete bias analysis.
        
        Args:
            reviews: List of review dictionaries with 'rating' and 'text'
            expected_rating_dist: Expected rating distribution from config
            
        Returns:
            Dictionary with all bias metrics
        """
        ratings = [r['rating'] for r in reviews]
        texts = [r['text'] for r in reviews]
        
        rating_analysis = self.analyze_rating_distribution(ratings, expected_rating_dist)
        sentiment_analysis = self.analyze_sentiment_consistency(reviews)
        pattern_analysis = self.detect_repetitive_patterns(texts)
        length_analysis = self.analyze_length_distribution(texts)
        
        # Overall bias assessment
        has_bias = (
            rating_analysis['is_biased'] or
            not sentiment_analysis['is_consistent'] or
            pattern_analysis['has_repetition_issues'] or
            length_analysis['is_too_uniform']
        )
        
        return {
            'rating_distribution': rating_analysis,
            'sentiment_consistency': sentiment_analysis,
            'repetitive_patterns': pattern_analysis,
            'length_distribution': length_analysis,
            'overall_bias_detected': has_bias
        }
