"""
Example script showing how to use the synthetic review generator.
"""
from dotenv import load_dotenv
from generate import generate_reviews
from compare import compare_reviews, print_statistics

# Load environment variables
load_dotenv()


def main():
    """Run example generation and analysis."""
    
    print("="*60)
    print("SYNTHETIC REVIEW GENERATOR - EXAMPLE")
    print("="*60)
    
    # Step 1: Generate synthetic reviews
    print("\n[1/3] Generating synthetic reviews...")
    reviews = generate_reviews(
        num_reviews=20,  # Small number for quick demo
        provider='gemini',  # Using Gemini (faster than Hugging Face for demo)
        config_path='config/generation_config.yaml',
        real_reviews_path='data/real_reviews.jsonl',
        output_path='data/generated_reviews.jsonl'
    )

    
    # Step 2: Show statistics
    print("\n[2/3] Displaying statistics...")
    print_statistics('data/generated_reviews.jsonl')
    
    # Step 3: Analyze quality
    print("\n[3/3] Analyzing quality...")
    report = compare_reviews(
        synthetic_path='data/generated_reviews.jsonl',
        real_path='data/real_reviews.jsonl',
        config_path='config/generation_config.yaml',
        output_path='quality_report.json'
    )
    
    print("\n" + "="*60)
    print("EXAMPLE COMPLETE!")
    print("="*60)
    print(f"\nGenerated {len(reviews)} reviews")
    print(f"Quality Score: {report['quality_score']['overall']}/100")
    print(f"Grade: {report['quality_score']['grade']}")
    print("\nCheck the following files:")
    print("  - data/generated_reviews.jsonl")
    print("  - quality_report.json")
    print("\n")


if __name__ == '__main__':
    main()
