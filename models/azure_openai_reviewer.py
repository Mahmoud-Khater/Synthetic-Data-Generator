"""
Azure OpenAI-based review reviewer.
"""
import os
from typing import Dict, List, Optional
from openai import AzureOpenAI


class AzureOpenAIReviewer:
    """Review reviewer using Azure OpenAI."""
    
    def __init__(self, config: Dict, api_key: Optional[str] = None, endpoint: Optional[str] = None, 
                 deployment: Optional[str] = None, api_version: Optional[str] = None):
        """
        Initialize Azure OpenAI reviewer.
        
        Args:
            config: Configuration dictionary
            api_key: Azure OpenAI API key (if None, reads from environment)
            endpoint: Azure OpenAI endpoint (if None, reads from environment)
            deployment: Deployment name (if None, reads from environment)
            api_version: API version (if None, reads from environment)
        """
        self.config = config
        self.review_params = config.get('review_params', {})
        
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
    
    def build_review_prompt(self, review_text: str, rating: int, persona: Dict) -> str:
        """
        Build a prompt for reviewing generated content.
        
        Args:
            review_text: The generated review text to evaluate
            rating: The rating (1-5) associated with the review
            persona: The persona dictionary used to generate the review
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are a quality assurance expert evaluating synthetic review data.

Review the following generated review and assess its quality:

**Generated Review:**
{review_text}

**Associated Rating:** {rating}/5
**Persona Used:**
- Demographics: {persona.get('demographics', 'N/A')}
- Preferences: {persona.get('preferences', 'N/A')}
- Writing Style: {persona.get('writing_style', 'N/A')}

**Evaluation Criteria:**
1. **Authenticity**: Does the review sound natural and human-written?
2. **Consistency**: Does the review content match the rating ({rating}/5)?
3. **Persona Alignment**: Does the review reflect the persona's characteristics?
4. **Quality**: Is the review well-written, coherent, and free of obvious errors?
5. **Realism**: Could this review plausibly appear on a real review platform?

**Provide your assessment in the following JSON format:**
{{
    "authenticity_score": <1-10>,
    "consistency_score": <1-10>,
    "persona_alignment_score": <1-10>,
    "quality_score": <1-10>,
    "realism_score": <1-10>,
    "overall_score": <1-10>,
    "pass": <true/false>,
    "issues": ["list of any issues found"],
    "suggestions": ["list of improvement suggestions"]
}}

Provide only the JSON output, no additional text."""
        
        return prompt
    
    def review_generated_content(self, review_text: str, rating: int, persona: Dict) -> Dict:
        """
        Review a generated review using Azure OpenAI API.
        
        Args:
            review_text: The generated review text to evaluate
            rating: The rating (1-5) associated with the review
            persona: The persona dictionary used to generate the review
            
        Returns:
            Dictionary containing review assessment
        """
        prompt = self.build_review_prompt(review_text, rating, persona)
        
        try:
            # Generate review assessment
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.review_params.get('temperature', 0.3),
                max_tokens=self.review_params.get('max_tokens', 1000),
                top_p=self.review_params.get('top_p', 0.9),
            )
            
            # Extract assessment
            assessment_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            import json
            # Remove markdown code blocks if present
            if assessment_text.startswith('```'):
                assessment_text = assessment_text.split('```')[1]
                if assessment_text.startswith('json'):
                    assessment_text = assessment_text[4:]
                assessment_text = assessment_text.strip()
            
            assessment = json.loads(assessment_text)
            
            return assessment
            
        except Exception as e:
            raise RuntimeError(f"Error reviewing content with Azure OpenAI: {str(e)}")
    
    def batch_review(self, reviews: List[Dict]) -> List[Dict]:
        """
        Review multiple generated reviews.
        
        Args:
            reviews: List of review dictionaries containing 'text', 'rating', and 'persona'
            
        Returns:
            List of assessment dictionaries
        """
        assessments = []
        
        for review in reviews:
            try:
                assessment = self.review_generated_content(
                    review_text=review['text'],
                    rating=review['rating'],
                    persona=review['persona']
                )
                assessment['original_review'] = review
                assessments.append(assessment)
            except Exception as e:
                assessments.append({
                    'error': str(e),
                    'original_review': review,
                    'pass': False
                })
        
        return assessments
    
    def get_provider_name(self) -> str:
        """Return the provider name."""
        return f"azure_openai_reviewer_{self.deployment}"
