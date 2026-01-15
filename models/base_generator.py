"""
Base generator interface for synthetic review generation.
All LLM providers should implement this interface.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import random


class BaseGenerator(ABC):
    """Abstract base class for review generators."""
    
    def __init__(self, config: Dict):
        """
        Initialize the generator with configuration.
        
        Args:
            config: Configuration dictionary from generation_config.yaml
        """
        self.config = config
        self.personas = config.get('personas', [])
        self.rating_distribution = config.get('rating_distribution', {})
        self.generation_params = config.get('generation_params', {})
        self.product_context = config.get('product_context', {})
        self.review_length = config.get('review_length', {})
    
    @abstractmethod
    def generate_review(self, rating: int, persona: Dict, real_reviews: List[str]) -> str:
        """
        Generate a single synthetic review.
        
        Args:
            rating: The rating (1-5) for this review
            persona: The persona dictionary to use for generation
            real_reviews: List of real reviews for context/style matching
            
        Returns:
            Generated review text
        """
        pass
    
    def select_rating(self) -> int:
        """
        Select a rating based on the configured distribution.
        
        Returns:
            Rating value (1-5)
        """
        ratings = list(self.rating_distribution.keys())
        weights = list(self.rating_distribution.values())
        return random.choices(ratings, weights=weights)[0]
    
    def select_persona(self) -> Dict:
        """
        Randomly select a persona from the configuration.
        
        Returns:
            Persona dictionary
        """
        return random.choice(self.personas)
    
    def build_prompt(self, rating: int, persona: Dict, real_reviews: List[str]) -> str:
        """
        Build the prompt for LLM generation.
        
        Args:
            rating: The rating (1-5) for this review
            persona: The persona dictionary to use
            real_reviews: Sample real reviews for style reference
            
        Returns:
            Formatted prompt string
        """
        # Sample a few real reviews for context
        sample_reviews = random.sample(real_reviews, min(5, len(real_reviews)))
        
        prompt = f"""You are writing a product review for {self.product_context.get('category', 'a product')}.

            Persona: {persona['name']}
            Description: {persona['description']}
            Traits:
            {chr(10).join(f"- {trait}" for trait in persona['traits'])}

            Rating: {rating}/5 stars

            Here are some example reviews for style reference:
            {chr(10).join(f'- "{review}"' for review in sample_reviews)}

            Write a realistic review from this persona's perspective with a {rating}/5 rating.
            The review should:
            - Be between {self.review_length.get('min_words', 20)} and {self.review_length.get('max_words', 150)} words
            - Match the persona's characteristics and perspective
            - Feel authentic and natural
            - Mention relevant aspects like: {', '.join(self.product_context.get('aspects', []))}
            - Match the rating appropriately (higher ratings = more positive)

            Write ONLY the review text, no additional commentary or labels."""

        return prompt
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of the LLM provider."""
        pass
