"""
Fixed sampling utilities for deterministic rating and trait selection.
Ensures exact distribution matching instead of random sampling.
"""
from typing import List, Dict, Any
import itertools


class FixedSampler:
    """Provides deterministic sampling for ratings and traits."""
    
    def __init__(self):
        """Initialize fixed sampler."""
        self.rating_cycle = None
        self.trait_cycles = {}
    
    def create_rating_sequence(self, rating_distribution: Dict[int, float], total_count: int) -> List[int]:
        """
        Create a fixed sequence of ratings matching the distribution exactly.
        
        Args:
            rating_distribution: Dict mapping rating (0-4) to probability
            total_count: Total number of ratings needed
            
        Returns:
            List of ratings in deterministic order
        """
        ratings = []
        
        # Calculate exact counts for each rating
        for rating in sorted(rating_distribution.keys()):
            probability = rating_distribution[rating]
            count = round(total_count * probability)
            ratings.extend([rating] * count)
        
        # Adjust if rounding caused mismatch
        while len(ratings) < total_count:
            # Add most common rating
            max_rating = max(rating_distribution.keys(), key=rating_distribution.get)
            ratings.append(max_rating)
        
        while len(ratings) > total_count:
            # Remove least common rating
            min_rating = min(rating_distribution.keys(), key=rating_distribution.get)
            ratings.remove(min_rating)
        
        return ratings
    
    def get_rating_sampler(self, rating_distribution: Dict[int, float], total_count: int):
        """
        Get a cycling iterator for ratings.
        
        Args:
            rating_distribution: Rating distribution config
            total_count: Total reviews to generate
            
        Returns:
            Iterator that cycles through ratings
        """
        if self.rating_cycle is None:
            sequence = self.create_rating_sequence(rating_distribution, total_count)
            self.rating_cycle = itertools.cycle(sequence)
        
        return self.rating_cycle
    
    def get_next_rating(self, rating_distribution: Dict[int, float], total_count: int) -> int:
        """
        Get next rating from fixed sequence.
        
        Args:
            rating_distribution: Rating distribution config
            total_count: Total reviews to generate
            
        Returns:
            Next rating in sequence
        """
        sampler = self.get_rating_sampler(rating_distribution, total_count)
        return next(sampler)
    
    def get_trait_sampler(self, persona_name: str, traits: List[str]):
        """
        Get a cycling iterator for persona traits.
        
        Args:
            persona_name: Name of persona
            traits: List of available traits
            
        Returns:
            Iterator that cycles through traits
        """
        if persona_name not in self.trait_cycles:
            self.trait_cycles[persona_name] = itertools.cycle(traits)
        
        return self.trait_cycles[persona_name]
    
    def get_next_trait(self, persona_name: str, traits: List[str]) -> str:
        """
        Get next trait from fixed sequence for this persona.
        
        Args:
            persona_name: Name of persona
            traits: List of available traits
            
        Returns:
            Next trait in sequence
        """
        sampler = self.get_trait_sampler(persona_name, traits)
        return next(sampler)
    
    def reset(self):
        """Reset all samplers."""
        self.rating_cycle = None
        self.trait_cycles = {}
