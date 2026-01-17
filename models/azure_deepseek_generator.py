"""
Azure DeepSeek Generator for synthetic review generation.
Uses DeepSeek-R1 model hosted on Azure OpenAI endpoint.
"""
from models.base_generator import BaseGenerator
from openai import OpenAI
import os
from typing import Dict, List


class AzureDeepSeekGenerator(BaseGenerator):
    """Generator using Azure-hosted DeepSeek-R1 model."""
    
    def __init__(self, config: Dict):
        """
        Initialize Azure DeepSeek generator.
        
        Args:
            config: Configuration dictionary with product_context, review_length, etc.
        """
        super().__init__(config)
        
        # Azure DeepSeek configuration
        self.endpoint = os.getenv('AZURE_DEEPSEEK_ENDPOINT')
        self.api_key = os.getenv('AZURE_DEEPSEEK_API_KEY')
        self.deployment_name = os.getenv('AZURE_DEEPSEEK_DEPLOYMENT')
        self.model_name = os.getenv('AZURE_DEEPSEEK_MODEL')
        
        if not self.api_key:
            raise ValueError("AZURE_DEEPSEEK_API_KEY environment variable not set")
        if not self.endpoint:
            raise ValueError("AZURE_DEEPSEEK_ENDPOINT environment variable not set")
        
        # Initialize OpenAI client with base_url (DeepSeek on Azure uses this schema)
        self.client = OpenAI(
            base_url=self.endpoint,
            api_key=self.api_key
        )
    
    def generate_review(self, rating: int, persona: Dict, real_reviews: List[Dict] = None) -> str:
        """
        Generate a review using Azure DeepSeek.
        
        Args:
            rating: Rating (0-4 scale)
            persona: Persona dictionary with name, description, traits
            real_reviews: Optional real reviews for context
            
        Returns:
            Generated review text
        """
        prompt = self.build_prompt(rating, persona, real_reviews)
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that writes authentic product reviews."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                temperature=0.9,
                max_tokens=250,
                timeout=60
            )
            
            review = response.choices[0].message.content.strip()
            return review
            
        except Exception as e:
            raise Exception(f"Azure DeepSeek generation failed: {str(e)}")
    
    def get_provider_name(self) -> str:
        """Return the provider name."""
        return f"azure_deepseek_{self.model_name}"
