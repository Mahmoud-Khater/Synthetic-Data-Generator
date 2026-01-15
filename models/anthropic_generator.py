"""
Anthropic Claude-based review generator.
"""
import os
from typing import Dict, List, Optional
from anthropic import Anthropic
from models.base_generator import BaseGenerator


class AnthropicGenerator(BaseGenerator):
    """Review generator using Anthropic's Claude models."""
    
    def __init__(self, config: Dict, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize Anthropic generator.
        
        Args:
            config: Configuration dictionary
            api_key: Anthropic API key (if None, reads from environment)
            model: Model name to use (if None, uses default from environment)
        """
        super().__init__(config)
        
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("Anthropic API key not provided and ANTHROPIC_API_KEY not set in environment")
        
        self.model = model or os.getenv('ANTHROPIC_MODEL', 'claude-3-5-sonnet-20241022')
        self.client = Anthropic(api_key=self.api_key)
    
    def generate_review(self, rating: int, persona: Dict, real_reviews: List[str]) -> str:
        """
        Generate a review using Anthropic's API.
        
        Args:
            rating: The rating (1-5) for this review
            persona: The persona dictionary to use
            real_reviews: List of real reviews for context
            
        Returns:
            Generated review text
        """
        prompt = self.build_prompt(rating, persona, real_reviews)
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.generation_params.get('max_tokens', 200),
                temperature=self.generation_params.get('temperature', 0.8),
                top_p=self.generation_params.get('top_p', 0.9),
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            review_text = response.content[0].text.strip()
            # Remove quotes if the model wrapped the review in them
            if review_text.startswith('"') and review_text.endswith('"'):
                review_text = review_text[1:-1]
            
            return review_text
            
        except Exception as e:
            raise RuntimeError(f"Error generating review with Anthropic: {str(e)}")
    
    def get_provider_name(self) -> str:
        """Return the provider name."""
        return f"anthropic_{self.model}"
