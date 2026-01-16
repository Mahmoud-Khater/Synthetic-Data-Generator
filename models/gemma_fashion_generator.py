"""
Gemma Fashion-based review generator using local model.
"""
import os
from typing import Dict, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from models.base_generator import BaseGenerator


class GemmaFashionGenerator(BaseGenerator):
    """Review generator using local gemma-fashion-tuner model."""
    
    def __init__(self, config: Dict, token: Optional[str] = None):
        """
        Initialize Gemma Fashion generator.
        
        Args:
            config: Configuration dictionary
            token: Hugging Face token (if needed, though using local model)
        """
        super().__init__(config)
        
        # Resolve path to local model
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'domain_models'))
        model_dir = os.path.join(base_path, 'models--luisastre--gemma-fashion-tuner', 'snapshots')
        
        try:
            # Get the actual snapshot directory (hash)
            self.model_path = next(os.path.join(model_dir, d) for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d)))
        except (StopIteration, FileNotFoundError):
             raise RuntimeError(f"Could not find model snapshot in {model_dir}")

        self.token = token or os.getenv('HUGGINGFACE_TOKEN')
        
        print(f"Loading Gemma Fashion model from: {self.model_path}...")
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            token=self.token,
            trust_remote_code=True
        )
        
        # Load model with appropriate settings
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            token=self.token,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # Create text generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        print(f"✓ Model loaded successfully!")
    
    def generate_review(self, rating: int, persona: Dict, real_reviews: List[str]) -> str:
        """
        Generate a review using Gemma Fashion model.
        
        Args:
            rating: The rating (0-4) for this review
            persona: The persona dictionary to use
            real_reviews: List of real reviews for context
            
        Returns:
            Generated review text
        """
        prompt = self.build_prompt(rating, persona, real_reviews)
        
        try:
            # Generate text
            outputs = self.generator(
                prompt,
                max_new_tokens=self.generation_params.get('max_tokens', 200),
                temperature=self.generation_params.get('temperature', 0.8),
                top_p=self.generation_params.get('top_p', 0.9),
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract generated text
            generated_text = outputs[0]['generated_text']
            
            # Remove the prompt from the output
            review_text = generated_text[len(prompt):].strip()
            
            # Clean up the review
            # Remove quotes if wrapped
            if review_text.startswith('"') and review_text.endswith('"'):
                review_text = review_text[1:-1]
            
            # Take only the first paragraph/sentence group
            lines = review_text.split('\n')
            review_text = lines[0].strip()
            
            # Limit length if too long
            words = review_text.split()
            max_words = self.review_length.get('max_words', 150)
            if len(words) > max_words:
                review_text = ' '.join(words[:max_words])
            
            return review_text
            
        except Exception as e:
            raise RuntimeError(f"Error generating review with Gemma Fashion: {str(e)}")
    
    def get_provider_name(self) -> str:
        """Return the provider name."""
        return "gemma_fashion"
