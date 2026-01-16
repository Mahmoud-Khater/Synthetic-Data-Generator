"""
Duplicate detection module using FAISS for semantic similarity.
Detects and removes duplicate reviews based on cosine similarity.
"""
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


class DuplicateDetector:
    """Detects and removes duplicate reviews using FAISS."""
    
    def __init__(self, similarity_threshold: float = 0.90):
        """
        Initialize duplicate detector.
        
        Args:
            similarity_threshold: Cosine similarity threshold (default: 0.90)
        """
        self.similarity_threshold = similarity_threshold
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def detect_duplicates(self, reviews: List[Dict]) -> Tuple[List[int], List[Dict]]:
        """
        Detect duplicate reviews using FAISS.
        
        Args:
            reviews: List of review dictionaries with 'text' field
            
        Returns:
            Tuple of (duplicate_indices, duplicate_pairs)
        """
        if len(reviews) < 2:
            return [], []
        
        # Extract texts
        texts = [r['text'] for r in reviews]
        
        # Generate embeddings
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors = cosine similarity
        index.add(embeddings)
        
        # Find duplicates
        duplicate_indices = set()
        duplicate_pairs = []
        
        for i in range(len(embeddings)):
            if i in duplicate_indices:
                continue
            
            # Search for similar reviews (k=10 to find potential duplicates)
            D, I = index.search(embeddings[i:i+1], min(10, len(embeddings)))
            
            for j, (similarity, idx) in enumerate(zip(D[0], I[0])):
                if idx == i:  # Skip self
                    continue
                
                if similarity >= self.similarity_threshold and idx not in duplicate_indices:
                    duplicate_indices.add(idx)
                    duplicate_pairs.append({
                        'index1': i,
                        'index2': int(idx),
                        'similarity': float(similarity),
                        'text1': texts[i][:100] + '...',
                        'text2': texts[int(idx)][:100] + '...'
                    })
        
        return list(duplicate_indices), duplicate_pairs
    
    def remove_duplicates(self, reviews: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        Remove duplicate reviews.
        
        Args:
            reviews: List of review dictionaries
            
        Returns:
            Tuple of (unique_reviews, stats)
        """
        duplicate_indices, duplicate_pairs = self.detect_duplicates(reviews)
        
        # Remove duplicates (keep first occurrence)
        unique_reviews = [r for i, r in enumerate(reviews) if i not in duplicate_indices]
        
        stats = {
            'original_count': len(reviews),
            'duplicate_count': len(duplicate_indices),
            'unique_count': len(unique_reviews),
            'duplicate_rate': len(duplicate_indices) / len(reviews) if reviews else 0,
            'duplicate_pairs': duplicate_pairs
        }
        
        return unique_reviews, stats
