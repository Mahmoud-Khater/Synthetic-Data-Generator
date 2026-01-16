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
            rating: The rating (0-4) for this review
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
            rating: The rating (0-4) for this review
            persona: The persona dictionary to use
            real_reviews: Sample real reviews for style reference
            
        Returns:
            Formatted prompt string
        """
        # Sample a few real reviews for context
        sample_reviews = random.sample(real_reviews, min(5, len(real_reviews)))
        examples_block = ""
        if sample_reviews:
            joined_examples = "\n".join(f'- "{r}"' for r in sample_reviews)
            examples_block = f"""
        Here are some example reviews for style reference:
        {joined_examples}
        """
        
        # Randomly select ONE trait from the persona's traits for more variety
        selected_trait = random.choice(persona['traits']) if persona.get('traits') else "casual reviewer"

        # Rating-specific tone guidance
        rating_guidance = {
            0: "Express clear disappointment and frustration. Use negative language.",
            1: "Show dissatisfaction. Mention specific problems and issues.",
            2: "Be balanced - mention both pros and cons. Use neutral, mixed language.",
            3: "Show satisfaction. Be positive but mention minor areas for improvement.",
            4: "Express enthusiasm and strong satisfaction. Use very positive language."
        }
        tone_instruction = rating_guidance.get(rating, "Match your tone to the rating.")

        prompt = f"""You are writing a product review for {self.product_context.get('category', 'a product')}.

            Persona: {persona['name']}
            Description: {persona['description']}
            Key Trait: {selected_trait}

            Rating: {rating}/4 stars

            {examples_block}

            Write a realistic review from this persona's perspective with a {rating}/4 rating.
            
            CRITICAL - Phrase Blacklist (DO NOT USE):
            - "the fit was" / "the fit is"
            - "for the price" / "for the price point"
            - "i expected" / "i was hoping"
            - "after just a" / "after a few"
            - "true to size"
            - "the style is" / "the style was"
            
            CRITICAL - Variety Requirements:
            - Start your review in a UNIQUE way (not "the", "for", "i was")
            - Use varied sentence structures throughout
            - Be creative with phrasing - avoid clichés
            - Make this review feel completely different from others
            
            Rating Alignment:
            - {tone_instruction}
            - Your language and sentiment MUST match the {rating}/4 rating
            
            The review should:
            - Be between {self.review_length.get('min_words', 20)} and {self.review_length.get('max_words', 150)} words
            - Reflect the key trait mentioned above
            - Feel authentic and natural
            - Mention relevant aspects like: {', '.join(self.product_context.get('aspects', []))}
            - Include specific personal context or use case

            Write ONLY the review text, no additional commentary or labels."""

        return prompt
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of the LLM provider."""
        pass
