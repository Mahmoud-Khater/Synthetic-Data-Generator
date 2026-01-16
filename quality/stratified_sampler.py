"""
Stratified sampler for real reviews.
Samples real reviews while preserving the original rating distribution.
"""
import json
from typing import List, Dict
from collections import defaultdict
import random


def load_real_reviews(filepath: str) -> List[Dict]:
    """
    Load all real reviews from JSONL file.
    
    Args:
        filepath: Path to JSONL file containing real reviews
        
    Returns:
        List of review dictionaries with 'text' and 'labels' (rating)
    """
    reviews = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            review = json.loads(line.strip())
            reviews.append(review)
    return reviews


def calculate_original_distribution(real_reviews: List[Dict]) -> Dict[int, float]:
    """
    Calculate rating distribution from all real reviews.
    
    Args:
        real_reviews: List of all real reviews
        
    Returns:
        Dictionary mapping rating to percentage (e.g., {1: 0.05, 2: 0.10, ...})
    """
    total = len(real_reviews)
    rating_counts = defaultdict(int)
    
    for review in real_reviews:
        rating = review.get('labels', review.get('rating', 3))
        rating_counts[rating] += 1
    
    # Convert counts to percentages
    distribution = {rating: count / total for rating, count in rating_counts.items()}
    
    return distribution


def stratified_sample(real_reviews: List[Dict], sample_size: int) -> List[Dict]:
    """
    Sample N reviews while preserving the original rating distribution.
    
    Args:
        real_reviews: List of all real reviews
        sample_size: Number of reviews to sample
        
    Returns:
        List of sampled reviews maintaining original distribution
    """
    # Calculate original distribution
    distribution = calculate_original_distribution(real_reviews)
    
    # Group reviews by rating
    reviews_by_rating = defaultdict(list)
    for review in real_reviews:
        rating = review.get('labels', review.get('rating', 3))
        reviews_by_rating[rating].append(review)
    
    # Sample from each rating group
    sampled_reviews = []
    
    for rating in sorted(reviews_by_rating.keys()):
        # Calculate how many to sample for this rating
        target_count = int(sample_size * distribution[rating])
        
        # Get available reviews for this rating
        available = reviews_by_rating[rating]
        
        # Sample (with replacement if needed)
        if len(available) >= target_count:
            sampled = random.sample(available, target_count)
        else:
            # If not enough reviews, sample with replacement
            sampled = random.choices(available, k=target_count)
        
        sampled_reviews.extend(sampled)
    
    # Shuffle to mix ratings
    random.shuffle(sampled_reviews)
    
    return sampled_reviews


def get_distribution_stats(reviews: List[Dict]) -> Dict:
    """
    Get statistics about the rating distribution.
    
    Args:
        reviews: List of reviews
        
    Returns:
        Dictionary with distribution statistics
    """
    total = len(reviews)
    rating_counts = defaultdict(int)
    
    for review in reviews:
        rating = review.get('labels', review.get('rating', 3))
        rating_counts[rating] += 1
    
    stats = {
        "total_reviews": total,
        "rating_counts": dict(rating_counts),
        "rating_percentages": {
            rating: (count / total * 100) for rating, count in rating_counts.items()
        }
    }
    
    return stats
