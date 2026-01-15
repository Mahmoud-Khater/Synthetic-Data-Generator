"""
DeepSeek-based review generator (OpenAI-compatible API).
"""
import os
from typing import Dict, List, Optional
from openai import OpenAI
from models.base_generator import BaseGenerator


class DeepSeekGenerator(BaseGenerator):
    """Review generator using DeepSeek API (OpenAI-compatible)."""
    
    def __init__(self, config: Dict, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """
        Initialize DeepSeek generator.
        
        Args:
            config: Configuration dictionary
            api_key: DeepSeek API key (if None, reads from environment)
            model_name: Model name to use (if None, uses default from environment)
        """
        super().__init__(config)
        
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError("DeepSeek API key not provided and DEEPSEEK_API_KEY not set in environment")
        
        self.model_name = model_name or os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')
        
        # Configure DeepSeek client (OpenAI-compatible)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )
    
    def generate_review(self, rating: int, persona: Dict, real_reviews: List[str]) -> str:
        """
        Generate a review using DeepSeek API.
        
        Args:
            rating: The rating (1-5) for this review
            persona: The persona dictionary to use
            real_reviews: List of real reviews for context
            
        Returns:
            Generated review text
        """
        prompt = self.build_prompt(rating, persona, real_reviews)
        
        try:
            # Generate content
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.generation_params.get('temperature', 0.8),
                max_tokens=self.generation_params.get('max_tokens', 1000),
                top_p=self.generation_params.get('top_p', 0.9),
            )
            
            # Extract text
            review_text = response.choices[0].message.content.strip()
            
            # Remove quotes if the model wrapped the review in them
            if review_text.startswith('"') and review_text.endswith('"'):
                review_text = review_text[1:-1]
            
            # Remove any markdown formatting
            review_text = review_text.replace('**', '').replace('*', '')
            
            return review_text
            
        except Exception as e:
            raise RuntimeError(f"Error generating review with DeepSeek: {str(e)}")
    
    def get_provider_name(self) -> str:
        """Return the provider name."""
        return f"deepseek_{self.model_name}"
