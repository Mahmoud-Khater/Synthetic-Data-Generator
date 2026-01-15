"""
Main generation script for synthetic reviews.
"""
import os
import json
import yaml
import pandas as pd
from typing import List, Dict
from tqdm import tqdm
from dotenv import load_dotenv
from models import OpenAIGenerator, AnthropicGenerator


def load_config(config_path: str = 'config/generation_config.yaml') -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_real_reviews(csv_path: str = 'data/real_reviews.csv') -> List[str]:
    """Load real reviews from CSV file."""
    df = pd.read_csv(csv_path)
    return df['review_text'].tolist()


def create_generator(provider: str, config: Dict):
    """
    Create a generator instance based on provider.
    
    Args:
        provider: Provider name ('openai' or 'anthropic')
        config: Configuration dictionary
        
    Returns:
        Generator instance
    """
    if provider.lower() == 'openai':
        return OpenAIGenerator(config)
    elif provider.lower() == 'anthropic':
        return AnthropicGenerator(config)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def generate_reviews(
    num_reviews: int,
    provider: str = 'openai',
    config_path: str = 'config/generation_config.yaml',
    real_reviews_path: str = 'data/real_reviews.csv',
    output_path: str = 'data/generated_reviews.jsonl'
) -> List[Dict]:
    """
    Generate synthetic reviews.
    
    Args:
        num_reviews: Number of reviews to generate
        provider: LLM provider to use
        config_path: Path to configuration file
        real_reviews_path: Path to real reviews CSV
        output_path: Path to save generated reviews
        
    Returns:
        List of generated review dictionaries
    """
    # Load configuration and real reviews
    config = load_config(config_path)
    real_reviews = load_real_reviews(real_reviews_path)
    
    # Create generator
    generator = create_generator(provider, config)
    
    print(f"\nGenerating {num_reviews} synthetic reviews using {generator.get_provider_name()}...")
    print(f"Real reviews loaded: {len(real_reviews)}")
    print(f"Personas available: {len(config['personas'])}")
    
    generated_reviews = []
    
    # Generate reviews with progress bar
    for i in tqdm(range(num_reviews), desc="Generating reviews"):
        # Select rating and persona
        rating = generator.select_rating()
        persona = generator.select_persona()
        
        try:
            # Generate review
            review_text = generator.generate_review(rating, persona, real_reviews)
            
            # Create review object
            review_obj = {
                'id': i + 1,
                'rating': rating,
                'text': review_text,
                'persona': persona['name'],
                'provider': generator.get_provider_name()
            }
            
            generated_reviews.append(review_obj)
            
            # Save incrementally (in case of interruption)
            if (i + 1) % 10 == 0:
                save_reviews(generated_reviews, output_path)
                
        except Exception as e:
            print(f"\nError generating review {i + 1}: {str(e)}")
            continue
    
    # Final save
    save_reviews(generated_reviews, output_path)
    
    print(f"\n✓ Successfully generated {len(generated_reviews)} reviews")
    print(f"✓ Saved to: {output_path}")
    
    return generated_reviews


def save_reviews(reviews: List[Dict], output_path: str):
    """Save reviews to JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for review in reviews:
            f.write(json.dumps(review, ensure_ascii=False) + '\n')


def load_generated_reviews(input_path: str = 'data/generated_reviews.jsonl') -> List[Dict]:
    """Load generated reviews from JSONL file."""
    reviews = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            reviews.append(json.loads(line))
    return reviews


if __name__ == '__main__':
    # Load environment variables
    load_dotenv()
    
    # Get provider from environment or use default
    provider = os.getenv('DEFAULT_PROVIDER', 'openai')
    
    # Load config to get num_reviews
    config = load_config()
    num_reviews = config.get('num_reviews', 100)
    
    # Generate reviews
    generate_reviews(
        num_reviews=num_reviews,
        provider=provider
    )
