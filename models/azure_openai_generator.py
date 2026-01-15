"""
Azure OpenAI-based review generator.
"""
import os
from typing import Dict, List, Optional
from openai import AzureOpenAI
from models.base_generator import BaseGenerator


class AzureOpenAIGenerator(BaseGenerator):
    """Review generator using Azure OpenAI."""
    
    def __init__(self, config: Dict, api_key: Optional[str] = None, endpoint: Optional[str] = None, 
                 deployment: Optional[str] = None, api_version: Optional[str] = None):
        """
        Initialize Azure OpenAI generator.
        
        Args:
            config: Configuration dictionary
            api_key: Azure OpenAI API key (if None, reads from environment)
            endpoint: Azure OpenAI endpoint (if None, reads from environment)
            deployment: Deployment name (if None, reads from environment)
            api_version: API version (if None, reads from environment)
        """
        super().__init__(config)
        
        self.api_key = api_key or os.getenv('AZURE_OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("Azure OpenAI API key not provided and AZURE_OPENAI_API_KEY not set in environment")
        
        self.endpoint = endpoint or os.getenv('AZURE_OPENAI_ENDPOINT')
        if not self.endpoint:
            raise ValueError("Azure OpenAI endpoint not provided and AZURE_OPENAI_ENDPOINT not set in environment")
        
        self.deployment = deployment or os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4o-mini')
        self.api_version = api_version or os.getenv('AZURE_OPENAI_API_VERSION', '2024-08-01-preview')
        
        # Configure Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint
        )
    
    def generate_review(self, rating: int, persona: Dict, real_reviews: List[str]) -> str:
        """
        Generate a review using Azure OpenAI API.
        
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
                model=self.deployment,  # This is the deployment name in Azure
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
            raise RuntimeError(f"Error generating review with Azure OpenAI: {str(e)}")
    
    def get_provider_name(self) -> str:
        """Return the provider name."""
        return f"azure_openai_{self.deployment}"
