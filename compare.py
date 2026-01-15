"""
Comparison script for real vs synthetic reviews.
"""
import pandas as pd
from typing import List, Dict
from generate import load_config, load_real_reviews, load_generated_reviews
from quality import QualityReporter


def load_real_reviews_with_ratings(csv_path: str = 'data/real_reviews.csv') -> List[Dict]:
    """
    Load real reviews with ratings from CSV.
    
    Args:
        csv_path: Path to real reviews CSV
        
    Returns:
        List of review dictionaries
    """
    df = pd.read_csv(csv_path)
    return [
        {'rating': row['rating'], 'text': row['review_text']}
        for _, row in df.iterrows()
    ]


def compare_reviews(
    synthetic_path: str = 'data/generated_reviews.jsonl',
    real_path: str = 'data/real_reviews.csv',
    config_path: str = 'config/generation_config.yaml',
    output_path: str = 'quality_report.json'
):
    """
    Compare synthetic reviews with real reviews and generate quality report.
    
    Args:
        synthetic_path: Path to generated reviews JSONL
        real_path: Path to real reviews CSV
        config_path: Path to configuration file
        output_path: Path to save quality report
    """
    print("\nLoading data...")
    
    # Load data
    config = load_config(config_path)
    synthetic_reviews = load_generated_reviews(synthetic_path)
    real_reviews = load_real_reviews(real_path)
    
    print(f"✓ Loaded {len(synthetic_reviews)} synthetic reviews")
    print(f"✓ Loaded {len(real_reviews)} real reviews")
    
    # Generate quality report
    print("\nAnalyzing quality metrics...")
    reporter = QualityReporter(config)
    report = reporter.generate_report(synthetic_reviews, real_reviews)
    
    # Save report
    reporter.save_report(report, output_path)
    print(f"✓ Report saved to: {output_path}")
    
    # Print summary
    reporter.print_summary(report)
    
    return report


def print_statistics(synthetic_path: str = 'data/generated_reviews.jsonl'):
    """
    Print basic statistics about generated reviews.
    
    Args:
        synthetic_path: Path to generated reviews JSONL
    """
    reviews = load_generated_reviews(synthetic_path)
    
    print("\n" + "="*60)
    print("SYNTHETIC REVIEWS STATISTICS")
    print("="*60)
    
    # Rating distribution
    from collections import Counter
    ratings = [r['rating'] for r in reviews]
    rating_counts = Counter(ratings)
    
    print("\nRating Distribution:")
    for rating in sorted(rating_counts.keys()):
        count = rating_counts[rating]
        percentage = (count / len(reviews)) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {rating} stars: {count:3d} ({percentage:5.1f}%) {bar}")
    
    # Persona distribution
    personas = [r['persona'] for r in reviews]
    persona_counts = Counter(personas)
    
    print("\nPersona Distribution:")
    for persona, count in persona_counts.most_common():
        percentage = (count / len(reviews)) * 100
        print(f"  {persona}: {count:3d} ({percentage:5.1f}%)")
    
    # Length statistics
    import numpy as np
    lengths = [len(r['text'].split()) for r in reviews]
    
    print("\nReview Length (words):")
    print(f"  Average: {np.mean(lengths):.1f}")
    print(f"  Median:  {np.median(lengths):.1f}")
    print(f"  Min:     {min(lengths)}")
    print(f"  Max:     {max(lengths)}")
    
    print("\n" + "="*60 + "\n")


if __name__ == '__main__':
    import sys
    
    # Print statistics
    print_statistics()
    
    # Run comparison
    compare_reviews()
