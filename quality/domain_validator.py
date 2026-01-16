"""
Domain validator for shoe reviews.
Validates that generated reviews are actually about shoes.
"""
from typing import Dict, List
import re


# Comprehensive list of shoe-related keywords
SHOE_KEYWORDS = {
    # Product types
    "shoe", "shoes", "sneaker", "sneakers", "boot", "boots", "sandal", "sandals",
    "loafer", "loafers", "heel", "heels", "flat", "flats", "slipper", "slippers",
    "footwear", "kicks", "trainers",
    
    # Fit and sizing
    "fit", "fits", "fitting", "size", "sizing", "sized", "true to size",
    "run small", "run large", "runs small", "runs large", "tight", "loose",
    "snug", "roomy", "narrow", "wide", "width",
    
    # Comfort features
    "comfort", "comfortable", "comfy", "cushion", "cushioning", "padding",
    "support", "supportive", "arch", "arch support", "insole", "insoles",
    "footbed", "midsole", "outsole",
    
    # Parts of shoe
    "sole", "soles", "toe", "toes", "toe box", "heel", "heels", "tongue",
    "laces", "lacing", "strap", "straps", "buckle", "zipper",
    "upper", "leather", "suede", "canvas", "mesh", "rubber",
    
    # Performance
    "traction", "grip", "slip", "slippery", "non-slip", "breathable",
    "ventilation", "waterproof", "durable", "durability", "wear",
    "break in", "breaking in", "broken in",
    
    # Activities
    "walking", "running", "jogging", "hiking", "gym", "workout",
    "basketball", "tennis", "athletic", "casual", "dress", "formal",
    "work", "office"
}


def validate_shoe_domain(review_text: str) -> Dict:
    """
    Check if a review is about shoes.
    
    Args:
        review_text: The review text to validate
        
    Returns:
        Dictionary with validation results:
        {
            "is_shoe_related": bool,
            "relevance_score": float (0-1),
            "detected_keywords": List[str],
            "keyword_count": int
        }
    """
    # Convert to lowercase for matching
    text_lower = review_text.lower()
    
    # Tokenize (simple word splitting)
    words = re.findall(r'\b\w+\b', text_lower)
    total_words = len(words)
    
    # Find matching keywords
    detected_keywords = []
    for keyword in SHOE_KEYWORDS:
        if keyword in text_lower:
            detected_keywords.append(keyword)
    
    keyword_count = len(detected_keywords)
    
    # Calculate relevance score
    # Score is based on: (unique keywords found / total words) * 100
    # But cap it at 1.0 for very keyword-dense reviews
    if total_words > 0:
        relevance_score = min(1.0, (keyword_count / total_words) * 10)
    else:
        relevance_score = 0.0
    
    # Determine if shoe-related
    # Threshold: at least 2 keywords or relevance score > 0.05
    is_shoe_related = keyword_count >= 2 or relevance_score > 0.05
    
    return {
        "is_shoe_related": is_shoe_related,
        "relevance_score": relevance_score,
        "detected_keywords": detected_keywords,
        "keyword_count": keyword_count
    }


def validate_reviews_batch(reviews: List[Dict]) -> Dict:
    """
    Validate multiple reviews for shoe domain relevance.
    
    Args:
        reviews: List of review dictionaries with 'text' field
        
    Returns:
        Dictionary with batch validation results
    """
    results = []
    flagged_reviews = []
    
    for i, review in enumerate(reviews):
        text = review.get('text', review.get('review', ''))
        validation = validate_shoe_domain(text)
        
        results.append(validation)
        
        if not validation['is_shoe_related']:
            flagged_reviews.append({
                "index": i,
                "text": text[:100] + "..." if len(text) > 100 else text,
                "relevance_score": validation['relevance_score'],
                "keyword_count": validation['keyword_count']
            })
    
    # Calculate aggregate statistics
    total = len(reviews)
    shoe_related_count = sum(1 for r in results if r['is_shoe_related'])
    avg_relevance = sum(r['relevance_score'] for r in results) / total if total > 0 else 0
    avg_keywords = sum(r['keyword_count'] for r in results) / total if total > 0 else 0
    
    return {
        "total_reviews": total,
        "shoe_related_count": shoe_related_count,
        "shoe_related_percentage": (shoe_related_count / total * 100) if total > 0 else 0,
        "avg_relevance_score": avg_relevance,
        "avg_keyword_count": avg_keywords,
        "flagged_reviews": flagged_reviews,
        "flagged_count": len(flagged_reviews)
    }
