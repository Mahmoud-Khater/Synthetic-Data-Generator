"""
Google Gemini-based review generator (using new google.genai library).
"""
import os
from typing import Dict, List, Optional
from google import genai
from google.genai import types
from models.base_generator import BaseGenerator


class GeminiGenerator(BaseGenerator):
    """Review generator using Google Gemini (new API)."""
    
    def __init__(self, config: Dict, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """
        Initialize Gemini generator.
        
        Args:
            config: Configuration dictionary
            api_key: Gemini API key (if None, reads from environment)
            model_name: Model name to use (if None, uses default from environment)
        """
        super().__init__(config)
        
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key not provided and GEMINI_API_KEY not set in environment")
        
        self.model_name = model_name or os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-exp')
        
        # Configure Gemini client
        self.client = genai.Client(api_key=self.api_key)
        
        # Configure generation parameters
        self.generation_config = types.GenerateContentConfig(
            temperature=self.generation_params.get('temperature', 0.8),
            top_p=self.generation_params.get('top_p', 0.9),
            max_output_tokens=self.generation_params.get('max_tokens', 500),
        )
    
    def generate_review(self, rating: int, persona: Dict, real_reviews: List[str]) -> str:
        """
        Generate a review using Gemini API.
        
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
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self.generation_config
            )
            
            # Extract text
            review_text = response.text.strip()
            
            # Remove quotes if the model wrapped the review in them
            if review_text.startswith('"') and review_text.endswith('"'):
                review_text = review_text[1:-1]
            
            # Remove any markdown formatting
            review_text = review_text.replace('**', '').replace('*', '')
            
            return review_text
            
        except Exception as e:
            raise RuntimeError(f"Error generating review with Gemini: {str(e)}")
    
    def get_provider_name(self) -> str:
        """Return the provider name."""
        return f"gemini_{self.model_name}"
