"""
Example script demonstrating the Azure OpenAI review generation workflow.
This script shows how to:
1. Generate reviews with quality checking
2. Regenerate up to 3 times if quality is poor
3. Generate a comprehensive quality report
"""
from build_graph import (
    run_review_generation,
    generate_batch_reviews,
    generate_quality_report,
    load_config
)


def example_single_review():
    """Generate a single review with quality checking."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Review Generation")
    print("="*80)
    
    # Load personas from config
    config = load_config()
    personas = config.get('personas', [])
    
    persona = personas[0]  # Use first persona
    rating = 4  # 0-4 scale
    
    result = run_review_generation(
        persona=persona,
        rating=rating,
        max_attempts=3
    )
    
    print("\n📝 Final Review:")
    print(f"   {result['review']}")
    print(f"\n📊 Quality Score: {result['quality_assessment'].get('overall_score', 0)}/10")
    print(f"✅ Passed: {result['quality_assessment'].get('pass', False)}")
    print(f"🔄 Attempts Used: {result['attempt']}/3")


def example_batch_generation():
    """Generate multiple reviews and create a quality report."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Batch Review Generation with Quality Report")
    print("="*80)
    
    # Load personas from config
    config = load_config()
    personas = config.get('personas', [])
    
    print(f"📋 Loaded {len(personas)} personas from config")
    
    # Generate reviews
    reviews = generate_batch_reviews(
        personas=personas,
        num_reviews=30,
        max_attempts=3
    )
    
    print(f"\n✅ Generated {len(reviews)} reviews")
    
    # Show summary statistics
    total_attempts = sum(r['attempts'] for r in reviews)
    passed_count = sum(1 for r in reviews if r['quality_assessment'].get('pass', False))
    
    print(f"\n📊 Generation Statistics:")
    print(f"   - Total reviews: {len(reviews)}")
    print(f"   - Passed quality check: {passed_count}/{len(reviews)}")
    print(f"   - Average attempts: {total_attempts/len(reviews):.2f}")
    
    # Generate quality report
    report = generate_quality_report(
        reviews=reviews,
        output_dir="reports"
    )
    
    return reviews, report


def example_custom_workflow():
    """Custom workflow with specific parameters."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Custom Workflow")
    print("="*80)
    
    # Generate reviews for specific rating distribution
    reviews = []
    
    # Generate 2 reviews for each rating
    for rating in [1, 2, 3, 4, 5]:
        for _ in range(2):
            import random
            persona = random.choice(EXAMPLE_PERSONAS)
            
            result = run_review_generation(
                persona=persona,
                rating=rating,
                max_attempts=3
            )
            
            reviews.append({
                "text": result["review"],
                "rating": result["rating"],
                "persona": result["persona"],
                "provider": "azure_openai",
                "attempts": result["attempt"],
                "quality_assessment": result["quality_assessment"]
            })
    
    print(f"\n✅ Generated {len(reviews)} reviews with balanced rating distribution")
    
    # Generate report
    report = generate_quality_report(
        reviews=reviews,
        output_dir="reports"
    )
    
    return reviews, report


if __name__ == "__main__":
    import sys
    
    print("\n" + "🚀"*40)
    print("Azure OpenAI Review Generation Workflow Demo")
    print("🚀"*40)
    
    # Choose which example to run
    if len(sys.argv) > 1:
        example = sys.argv[1]
    else:
        print("\nAvailable examples:")
        print("  1. Single review generation")
        print("  2. Batch generation with quality report (default)")
        print("  3. Custom workflow with balanced ratings")
        
        example = input("\nSelect example (1-3, default=2): ").strip() or "2"
    
    if example == "1":
        example_single_review()
    elif example == "2":
        example_batch_generation()
    elif example == "3":
        example_custom_workflow()
    else:
        print(f"Unknown example: {example}")
        sys.exit(1)
    
    print("\n✅ Demo complete!")
