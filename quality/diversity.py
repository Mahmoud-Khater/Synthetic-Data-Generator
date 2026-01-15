"""
Diversity metrics for synthetic reviews.
Measures vocabulary overlap and semantic similarity.
"""
from typing import List, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from collections import Counter

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class DiversityAnalyzer:
    """Analyzes diversity of synthetic reviews."""
    
    def __init__(self):
        """Initialize the diversity analyzer."""
        self.embedding_model = None
    
    def _load_embedding_model(self):
        """Lazy load the sentence transformer model."""
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        return self.embedding_model
    
    def calculate_vocabulary_overlap(self, reviews: List[str]) -> Dict[str, float]:
        """
        Calculate vocabulary diversity metrics.
        
        Args:
            reviews: List of review texts
            
        Returns:
            Dictionary with vocabulary metrics
        """
        # Tokenize all reviews
        all_tokens = []
        review_tokens = []
        
        for review in reviews:
            tokens = nltk.word_tokenize(review.lower())
            tokens = [t for t in tokens if t.isalnum()]  # Keep only alphanumeric
            review_tokens.append(tokens)
            all_tokens.extend(tokens)
        
        # Calculate metrics
        total_tokens = len(all_tokens)
        unique_tokens = len(set(all_tokens))
        
        # Type-Token Ratio (TTR)
        ttr = unique_tokens / total_tokens if total_tokens > 0 else 0
        
        # Average unique tokens per review
        avg_unique_per_review = np.mean([len(set(tokens)) for tokens in review_tokens])
        
        # Token frequency distribution
        token_freq = Counter(all_tokens)
        top_10_freq = sum([count for _, count in token_freq.most_common(10)])
        top_10_ratio = top_10_freq / total_tokens if total_tokens > 0 else 0
        
        return {
            'type_token_ratio': round(ttr, 4),
            'unique_tokens': unique_tokens,
            'total_tokens': total_tokens,
            'avg_unique_per_review': round(avg_unique_per_review, 2),
            'top_10_token_ratio': round(top_10_ratio, 4)
        }
    
    def calculate_semantic_similarity(self, reviews: List[str], sample_size: int = 100) -> Dict[str, float]:
        """
        Calculate semantic similarity between reviews using embeddings.
        
        Args:
            reviews: List of review texts
            sample_size: Number of review pairs to sample for efficiency
            
        Returns:
            Dictionary with similarity metrics
        """
        if len(reviews) < 2:
            return {'avg_similarity': 0.0, 'max_similarity': 0.0, 'min_similarity': 0.0}
        
        # Load model and generate embeddings
        model = self._load_embedding_model()
        embeddings = model.encode(reviews)
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(embeddings)
        
        # Get upper triangle (excluding diagonal)
        n = len(reviews)
        upper_triangle_indices = np.triu_indices(n, k=1)
        similarity_values = similarities[upper_triangle_indices]
        
        # Sample if too many pairs
        if len(similarity_values) > sample_size:
            similarity_values = np.random.choice(similarity_values, sample_size, replace=False)
        
        return {
            'avg_similarity': round(float(np.mean(similarity_values)), 4),
            'max_similarity': round(float(np.max(similarity_values)), 4),
            'min_similarity': round(float(np.min(similarity_values)), 4),
            'std_similarity': round(float(np.std(similarity_values)), 4)
        }
    
    def calculate_tfidf_diversity(self, reviews: List[str]) -> Dict[str, float]:
        """
        Calculate diversity using TF-IDF vectors.
        
        Args:
            reviews: List of review texts
            
        Returns:
            Dictionary with TF-IDF diversity metrics
        """
        if len(reviews) < 2:
            return {'avg_tfidf_similarity': 0.0}
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(reviews)
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(tfidf_matrix)
        
        # Get upper triangle (excluding diagonal)
        n = len(reviews)
        upper_triangle_indices = np.triu_indices(n, k=1)
        similarity_values = similarities[upper_triangle_indices]
        
        return {
            'avg_tfidf_similarity': round(float(np.mean(similarity_values)), 4),
            'std_tfidf_similarity': round(float(np.std(similarity_values)), 4)
        }
    
    def analyze(self, reviews: List[str]) -> Dict[str, any]:
        """
        Perform complete diversity analysis.
        
        Args:
            reviews: List of review texts
            
        Returns:
            Dictionary with all diversity metrics
        """
        vocab_metrics = self.calculate_vocabulary_overlap(reviews)
        semantic_metrics = self.calculate_semantic_similarity(reviews)
        tfidf_metrics = self.calculate_tfidf_diversity(reviews)
        
        return {
            'vocabulary': vocab_metrics,
            'semantic_similarity': semantic_metrics,
            'tfidf_similarity': tfidf_metrics
        }
