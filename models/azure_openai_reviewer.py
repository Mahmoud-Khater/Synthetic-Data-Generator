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
            rating: The rating (0-4) associated with the review
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
1. **Authenticity**: Does the review sound genuinely human-written with natural imperfections? Should feel personal and unique, not templated.
2. **Consistency**: Does the review content STRONGLY match the rating ({rating}/4)? Content must clearly justify the rating with specific examples.
3. **Persona Alignment**: Does the review clearly reflect the persona's perspective and traits? Generic reviews score lower.
4. **Quality**: Is the review detailed, well-structured, and includes MULTIPLE specific examples? Vague or generic statements score poorly.
5. **Realism**: Does this feel like a real person sharing their genuine experience? Should mention specific use cases and personal context.

**Strict Scoring Guidelines:**
- Reviews MUST include at least 3-4 specific product details (fit, comfort, materials, style, durability, price, etc.)
- Reviews should include personal context or use case (e.g., "wore them to work", "used for running", "bought for my son")
- Generic statements like "good quality" or "comfortable" without specifics score LOW (5-6/10)
- Reviews should have natural variation - not all reviews should follow the same structure
- Rating alignment is CRITICAL: 4-star must be enthusiastic, 0-star must express clear disappointment
- Overly polished or marketing-like language scores lower (7/10 max)
- Reviews lacking specific examples or personal experience score 6-7/10
- If persona details are N/A or minimal, give neutral scores (7-8) for persona alignment only
- Score each criterion from 1-10
- Calculate overall_score as the average of all criteria scores
- Set "pass" to true if overall_score is 7.5 or higher
- Set "pass" to false if overall_score is below 7.5

**What makes an EXCELLENT review (8+):**
- Mentions 3+ specific product features with details
- Includes personal story, use case, or context
- Has natural language with some imperfections (not overly polished)
- Strongly matches the rating with clear justification
- Feels unique and genuine, not generic or templated
- Shows the persona's perspective clearly

**What makes a POOR review (below 7.5):**
- Generic statements without specifics ("nice shoes", "good quality")
- No personal context or use case mentioned
- Doesn't clearly justify the rating
- Feels templated or too similar to other reviews
- Lacks detail or specific examples
- Too short or incomplete

**Provide your assessment in the following JSON format:**
{{
    "authenticity_score": <1-10>,
    "consistency_score": <1-10>,
    "persona_alignment_score": <1-10>,
    "quality_score": <1-10>,
    "realism_score": <1-10>,
    "overall_score": <1-10>,
    "pass": <true if overall_score >= 7.5, false otherwise>,
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
            rating: The rating (0-4) associated with the review
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
