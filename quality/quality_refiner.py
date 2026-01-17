"""
Quality refinement module for iterative improvement.
Identifies problematic reviews and regenerates them to improve overall quality.
"""
from typing import List, Dict, Tuple
import random
from collections import Counter


class QualityRefiner:
    """Refines review quality through iterative improvement."""
    
    def __init__(self):
        """Initialize quality refiner."""
        pass
    
    def identify_problematic_reviews(self, reviews: List[Dict], quality_report: Dict) -> Tuple[List[int], Dict]:
        """
        Identify reviews that are causing quality issues and analyze failure reasons.
        
        Args:
            reviews: List of review dictionaries
            quality_report: Quality report with bias and diversity metrics
            
        Returns:
            Tuple of (problematic_indices, failure_analysis)
        """
        problematic_indices = set()
        failure_reasons = {
            'repetitive_phrases': [],
            'sentiment_mismatch': [],
            'too_short': [],
            'too_long': [],
            'low_diversity': []
        }
        
        # 1. Find reviews with repetitive phrases
        repetitive_phrases = quality_report.get('bias', {}).get('repetitive_patterns', {}).get('repetitive_phrases', [])
        
        for phrase_info in repetitive_phrases[:5]:  # Top 5 most repetitive
            phrase = phrase_info['phrase']
            frequency = phrase_info['frequency']
            
            # If phrase appears in >20% of reviews, remove some instances
            if frequency > 0.2:
                for i, review in enumerate(reviews):
                    if phrase in review['text'].lower():
                        problematic_indices.add(i)
                        failure_reasons['repetitive_phrases'].append({
                            'index': i,
                            'phrase': phrase,
                            'frequency': frequency
                        })
                        # Only remove up to 30% of reviews with this phrase
                        if len(problematic_indices) >= len(reviews) * 0.3:
                            break
        
        # 2. Find reviews with sentiment-rating mismatches
        sentiment_issues = quality_report.get('bias', {}).get('sentiment_consistency', {}).get('examples', [])
        
        for issue in sentiment_issues:
            rating = issue['rating']
            # Find reviews with this rating that might be problematic
            for i, review in enumerate(reviews):
                if review.get('rating') == rating and i not in problematic_indices:
                    # Check if this specific review has the issue
                    text = review['text'].lower()
                    pos_words = ['great', 'excellent', 'amazing', 'perfect', 'love', 'best']
                    neg_words = ['bad', 'terrible', 'worst', 'hate', 'disappointed', 'poor']
                    
                    pos_count = sum(1 for word in pos_words if word in text)
                    neg_count = sum(1 for word in neg_words if word in text)
                    
                    if (rating >= 3 and neg_count > pos_count) or (rating <= 1 and pos_count > neg_count):
                        problematic_indices.add(i)
                        failure_reasons['sentiment_mismatch'].append({
                            'index': i,
                            'rating': rating,
                            'pos_count': pos_count,
                            'neg_count': neg_count
                        })
                        break
        
        # 3. Find reviews that are too short or too long
        for i, review in enumerate(reviews):
            word_count = len(review['text'].split())
            if word_count < 5:
                problematic_indices.add(i)
                failure_reasons['too_short'].append({'index': i, 'word_count': word_count})
            elif word_count > 150:
                problematic_indices.add(i)
                failure_reasons['too_long'].append({'index': i, 'word_count': word_count})
        
        # 3. Find very similar reviews (high semantic similarity outliers)
        # This would require embeddings - skip for now or implement if needed
        
        # Limit to removing max 40% of reviews per iteration
        max_remove = int(len(reviews) * 0.4)
        problematic_indices = list(problematic_indices)[:max_remove]
        
        # Create failure analysis summary
        failure_analysis = {
            'total_problematic': len(problematic_indices),
            'reasons': failure_reasons,
            'indices': problematic_indices
        }
        
        return problematic_indices, failure_analysis
    
    def refine_reviews(
        self, 
        reviews: List[Dict], 
        quality_report: Dict,
        generator,
        reviewer,
        personas: List[Dict],
        rating_distribution: Dict,
        real_reviews: List[Dict],
        max_attempts: int = 3
    ) -> Tuple[List[Dict], Dict, int]:
        """
        Refine reviews by analyzing failures, removing problematic ones, and regenerating.
        
        Args:
            reviews: Current reviews
            quality_report: Quality report
            generator: Generator instance
            reviewer: Reviewer instance
            personas: Available personas
            rating_distribution: Rating distribution config
            real_reviews: Real reviews for context
            max_attempts: Max regeneration attempts per review
            
        Returns:
            Tuple of (refined_reviews, improvement_stats, removed_count)
        """
        # Identify problematic reviews with failure analysis
        problematic_indices, failure_analysis = self.identify_problematic_reviews(reviews, quality_report)
        
        if not problematic_indices:
            return reviews, {'removed': 0, 'regenerated': 0}, 0
        
        print(f"\n🔧 Quality Refinement - Failure Analysis:")
        print(f"   - Total problematic reviews: {len(problematic_indices)}")
        
        # Print failure reasons
        reasons = failure_analysis['reasons']
        if reasons['repetitive_phrases']:
            print(f"   - Repetitive phrases: {len(reasons['repetitive_phrases'])} reviews")
        if reasons['sentiment_mismatch']:
            print(f"   - Sentiment mismatches: {len(reasons['sentiment_mismatch'])} reviews")
        if reasons['too_short']:
            print(f"   - Too short: {len(reasons['too_short'])} reviews")
        if reasons['too_long']:
            print(f"   - Too long: {len(reasons['too_long'])} reviews")
        
        print(f"   - Removing and regenerating...")
        
        # Keep good reviews
        good_reviews = [r for i, r in enumerate(reviews) if i not in problematic_indices]
        
        # Regenerate problematic ones
        regenerated_count = 0
        
        for idx in problematic_indices:
            old_review = reviews[idx]
            persona = old_review.get('persona', random.choice(personas))
            rating = old_review.get('rating', self._select_rating(rating_distribution))
            
            # Try to generate a better review
            from build_graph import run_review_generation
            
            result = run_review_generation(persona, rating, max_attempts)
            
            new_review = {
                "text": result["review"],
                "rating": result["rating"],
                "persona": result["persona"],
                "model": result.get("generator_used", generator.get_provider_name()),  # Use tracked generator
                "attempts": result["attempt"],
                "quality_assessment": result["quality_assessment"],
                "regenerated": True  # Mark as regenerated
            }
            
            good_reviews.append(new_review)
            regenerated_count += 1
        
        stats = {
            'removed': len(problematic_indices),
            'regenerated': regenerated_count,
            'failure_analysis': failure_analysis
        }
        
        return good_reviews, stats, len(problematic_indices)
    
    def _select_rating(self, rating_distribution: Dict) -> int:
        """Select a rating based on distribution."""
        ratings = list(rating_distribution.keys())
        weights = list(rating_distribution.values())
        return random.choices(ratings, weights=weights, k=1)[0]
