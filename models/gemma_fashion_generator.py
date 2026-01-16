"""
Gemma Fashion-based review generator using local model.
"""
import os
from typing import Dict, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoConfig
from models.base_generator import BaseGenerator

class GemmaFashionGenerator(BaseGenerator):
    
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
        self.model = None
        self.tokenizer = None
        self.generator = None

    def load(self):
        """Load the model if not already loaded."""
        if self.model is not None:
            return

        print(f"Loading Gemma Fashion model from: {self.model_path}...")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            token=self.token,
            use_fast=True,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with explicit CPU settings as requested
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            token=self.token,
            device_map=None,        # CPU only
            torch_dtype=torch.float32,  # CPU friendly
            trust_remote_code=True
        )
        
        self.model.to("cpu")
        self.model.eval()
        
        print(f"✓ Gemma Model loaded successfully (CPU mode)!")

    def unload(self):
        """Unload model and free memory."""
        if self.model is not None:
            print("Unloading Gemma Model...")
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("✓ Gemma Model unloaded!")

    def generate_review(self, rating: int, persona: Dict, real_reviews: List[str]) -> str:
        """
        Generate a review using Gemma Fashion model.
        """
        self.load()
        prompt = self.build_prompt(rating, persona, real_reviews)
        print(prompt)
        try:
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = inputs.to("cpu") # Ensure inputs are on CPU
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.generation_params.get('max_tokens', 200),
                    do_sample=True,
                    temperature=self.generation_params.get('temperature', 0.7),
                    top_p=self.generation_params.get('top_p', 0.9),
                    repetition_penalty=1.1,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from the output
            # gen_only = outputs[0][inputs["input_ids"].shape[-1]:]
            # review_text = self.tokenizer.decode(gen_only, skip_special_tokens=True).strip()
            review_text = generated_text
            # Clean up the review
            # Remove quotes if wrapped
            # if review_text.startswith('"') and review_text.endswith('"'):
            #     review_text = review_text[1:-1]
            
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
