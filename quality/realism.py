"""
Realism metrics for synthetic reviews.
Checks domain-specific realism and authenticity.
"""
from typing import List, Dict
import re
import textstat


class RealismAnalyzer:
    """Analyzes realism of synthetic reviews."""
    
    def __init__(self, product_context: Dict):
        """
        Initialize the realism analyzer.
        
        Args:
            product_context: Product context from config
        """
        self.product_context = product_context
        self.category = product_context.get('category', 'product')
        self.aspects = product_context.get('aspects', [])
    
    def analyze_aspect_coverage(self, reviews: List[str]) -> Dict[str, any]:
        """
        Check if reviews mention relevant product aspects.
        
        Args:
            reviews: List of review texts
            
        Returns:
            Dictionary with aspect coverage metrics
        """
        aspect_mentions = {aspect: 0 for aspect in self.aspects}
        reviews_with_aspects = 0
        
        for review in reviews:
            review_lower = review.lower()
            mentioned_aspects = []
            
            for aspect in self.aspects:
                if aspect.lower() in review_lower:
                    aspect_mentions[aspect] += 1
                    mentioned_aspects.append(aspect)
            
            if mentioned_aspects:
                reviews_with_aspects += 1
        
        total_reviews = len(reviews)
        coverage_rate = reviews_with_aspects / total_reviews if total_reviews > 0 else 0
        
        return {
            'aspect_mentions': aspect_mentions,
            'reviews_with_aspects': reviews_with_aspects,
            'total_reviews': total_reviews,
            'coverage_rate': round(coverage_rate, 4),
            'avg_mentions_per_aspect': round(sum(aspect_mentions.values()) / len(self.aspects), 2) if self.aspects else 0,
            'has_good_coverage': coverage_rate > 0.6
        }
    
    def analyze_readability(self, reviews: List[str]) -> Dict[str, any]:
        """
        Analyze readability scores to ensure natural language.
        
        Args:
            reviews: List of review texts
            
        Returns:
            Dictionary with readability metrics
        """
        flesch_scores = []
        
        for review in reviews:
            try:
                score = textstat.flesch_reading_ease(review)
                flesch_scores.append(score)
            except:
                continue
        
        if not flesch_scores:
            return {'error': 'Could not calculate readability scores'}
        
        avg_score = sum(flesch_scores) / len(flesch_scores)
        
        # Flesch Reading Ease interpretation
        # 90-100: Very Easy (5th grade)
        # 80-90: Easy (6th grade)
        # 70-80: Fairly Easy (7th grade)
        # 60-70: Standard (8th-9th grade)
        # 50-60: Fairly Difficult (10th-12th grade)
        
        return {
            'avg_flesch_score': round(avg_score, 2),
            'min_score': round(min(flesch_scores), 2),
            'max_score': round(max(flesch_scores), 2),
            'interpretation': self._interpret_flesch_score(avg_score),
            'is_natural': 50 <= avg_score <= 90  # Reasonable range for reviews
        }
    
    def _interpret_flesch_score(self, score: float) -> str:
        """Interpret Flesch Reading Ease score."""
        if score >= 90:
            return "Very Easy"
        elif score >= 80:
            return "Easy"
        elif score >= 70:
            return "Fairly Easy"
        elif score >= 60:
            return "Standard"
        elif score >= 50:
            return "Fairly Difficult"
        else:
            return "Difficult"
    
    def detect_ai_patterns(self, reviews: List[str]) -> Dict[str, any]:
        """
        Detect common AI-generated text patterns.
        
        Args:
            reviews: List of review texts
            
        Returns:
            Dictionary with AI pattern detection results
        """
        # Common AI phrases that might indicate synthetic text
        ai_indicators = [
            r'\bas an ai\b',
            r'\bi cannot\b',
            r'\bi don\'t have personal\b',
            r'\bdelve into\b',
            r'\bin conclusion\b',
            r'\bin summary\b',
            r'\bto summarize\b',
            r'\boverall experience\b.*\bwas\b',
        ]
        
        reviews_with_ai_patterns = 0
        pattern_matches = []
        
        for review in reviews:
            review_lower = review.lower()
            for pattern in ai_indicators:
                if re.search(pattern, review_lower):
                    reviews_with_ai_patterns += 1
                    pattern_matches.append({
                        'review_snippet': review[:100] + '...',
                        'pattern': pattern
                    })
                    break
        
        total_reviews = len(reviews)
        ai_pattern_rate = reviews_with_ai_patterns / total_reviews if total_reviews > 0 else 0
        
        return {
            'reviews_with_ai_patterns': reviews_with_ai_patterns,
            'total_reviews': total_reviews,
            'ai_pattern_rate': round(ai_pattern_rate, 4),
            'is_realistic': ai_pattern_rate < 0.05,  # Less than 5% is acceptable
            'examples': pattern_matches[:3]
        }
    
    def analyze_personal_pronouns(self, reviews: List[str]) -> Dict[str, any]:
        """
        Analyze use of personal pronouns (realistic reviews use "I", "my", etc.).
        
        Args:
            reviews: List of review texts
            
        Returns:
            Dictionary with pronoun usage metrics
        """
        pronouns = ['i ', 'my ', 'me ', 'i\'m', 'i\'ve', 'i\'ll']
        reviews_with_pronouns = 0
        
        for review in reviews:
            review_lower = ' ' + review.lower() + ' '
            if any(pronoun in review_lower for pronoun in pronouns):
                reviews_with_pronouns += 1
        
        total_reviews = len(reviews)
        pronoun_rate = reviews_with_pronouns / total_reviews if total_reviews > 0 else 0
        
        return {
            'reviews_with_pronouns': reviews_with_pronouns,
            'total_reviews': total_reviews,
            'pronoun_usage_rate': round(pronoun_rate, 4),
            'is_natural': pronoun_rate > 0.5  # Most real reviews use first person
        }
    
    def analyze(self, reviews: List[str]) -> Dict[str, any]:
        """
        Perform complete realism analysis.
        
        Args:
            reviews: List of review texts
            
        Returns:
            Dictionary with all realism metrics
        """
        aspect_analysis = self.analyze_aspect_coverage(reviews)
        readability_analysis = self.analyze_readability(reviews)
        ai_pattern_analysis = self.detect_ai_patterns(reviews)
        pronoun_analysis = self.analyze_personal_pronouns(reviews)
        
        # Overall realism assessment
        is_realistic = (
            aspect_analysis.get('has_good_coverage', False) and
            readability_analysis.get('is_natural', False) and
            ai_pattern_analysis.get('is_realistic', False) and
            pronoun_analysis.get('is_natural', False)
        )
        
        return {
            'aspect_coverage': aspect_analysis,
            'readability': readability_analysis,
            'ai_patterns': ai_pattern_analysis,
            'pronoun_usage': pronoun_analysis,
            'overall_realistic': is_realistic
        }
