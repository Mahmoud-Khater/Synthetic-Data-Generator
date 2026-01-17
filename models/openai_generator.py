"""
OpenAI-based review generator.
"""
import os
from typing import Dict, List
from openai import OpenAI
from models.base_generator import BaseGenerator


class OpenAIGenerator(BaseGenerator):
    """Review generator using OpenAI's GPT models."""
    
    def __init__(self, config: Dict, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize OpenAI generator.
        
        Args:
            config: Configuration dictionary
            api_key: OpenAI API key (if None, reads from environment)
            model: Model name to use (if None, uses default from environment)
        """
        super().__init__(config)
        
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY not set in environment")
        
        self.model = model or os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
        self.client = OpenAI(api_key=self.api_key)
    
    def generate_review(self, rating: int, persona: Dict, real_reviews: List[str]) -> str:
        """
        Generate a review using OpenAI's API.
        
        Args:
            rating: The rating (0-4) for this review
            persona: The persona dictionary to use
            real_reviews: List of real reviews for context
            
        Returns:
            Generated review text
        """
        prompt = self.build_prompt(rating, persona, real_reviews)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that writes realistic product reviews. Write only the review text without any additional commentary."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.generation_params.get('temperature', 0.8),
                max_tokens=self.generation_params.get('max_tokens', 200),
                top_p=self.generation_params.get('top_p', 0.9)
            )
            
            review_text = response.choices[0].message.content.strip()
            # Remove quotes if the model wrapped the review in them
            if review_text.startswith('"') and review_text.endswith('"'):
                review_text = review_text[1:-1]
            
            return review_text
            
        except Exception as e:
            raise RuntimeError(f"Error generating review with OpenAI: {str(e)}")
    
    def get_provider_name(self) -> str:
        """Return the provider name."""
        return f"openai_{self.model}"
