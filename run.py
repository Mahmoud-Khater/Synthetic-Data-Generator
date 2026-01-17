#!/usr/bin/env python3
"""
Simple CLI to generate synthetic reviews.
Usage: python run.py [num_reviews]
Default: 30 reviews
"""
import sys
from build_graph import (
    generate_batch_reviews,
    generate_quality_report,
    load_config
)


def main():
    """Generate reviews and quality report."""
    # Load config
    config = load_config()
    
    # Get number of reviews from command line or use config default
    default_num_reviews = config.get('num_reviews', 30)
    num_reviews = int(sys.argv[1]) if len(sys.argv) > 1 else default_num_reviews
    
    print(f"\n🚀 Generating {num_reviews} synthetic reviews...\n")
    
    # Load personas from config
    personas = config.get('personas', [])
    max_attempts = config.get('max_attempts', 3)
    
    print(f"📋 Loaded {len(personas)} personas from config")
    print(f"🔄 Max attempts per review: {max_attempts}")
    
    # Generate reviews
    reviews = generate_batch_reviews(
        personas=personas,
        num_reviews=num_reviews,
        max_attempts=max_attempts
    )
    
    print(f"\n✅ Generated {len(reviews)} reviews")
    
    # Generate quality report
    print(f"\n📊 Generating quality report...")
    report = generate_quality_report(reviews)
    
    print(f"\n✨ Done! Check the reports/ folder for results.")


if __name__ == "__main__":
    main()
