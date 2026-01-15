"""
Quality report generation for synthetic reviews.
Combines all quality metrics into a comprehensive report.
"""
from typing import Dict, List
import json
from datetime import datetime
from quality.diversity import DiversityAnalyzer
from quality.bias import BiasAnalyzer
from quality.realism import RealismAnalyzer


class QualityReporter:
    """Generates comprehensive quality reports for synthetic reviews."""
    
    def __init__(self, config: Dict):
        """
        Initialize the quality reporter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.diversity_analyzer = DiversityAnalyzer()
        self.bias_analyzer = BiasAnalyzer()
        self.realism_analyzer = RealismAnalyzer(config.get('product_context', {}))
    
    def generate_report(self, synthetic_reviews: List[Dict], real_reviews: List[str] = None) -> Dict:
        """
        Generate a comprehensive quality report.
        
        Args:
            synthetic_reviews: List of synthetic review dictionaries
            real_reviews: Optional list of real reviews for comparison
            
        Returns:
            Dictionary containing the complete quality report
        """
        synthetic_texts = [r['text'] for r in synthetic_reviews]
        
        # Run all analyses
        diversity_results = self.diversity_analyzer.analyze(synthetic_texts)
        bias_results = self.bias_analyzer.analyze(
            synthetic_reviews,
            self.config.get('rating_distribution', {})
        )
        realism_results = self.realism_analyzer.analyze(synthetic_texts)
        
        # Compare with real reviews if provided
        comparison = None
        if real_reviews:
            comparison = self._compare_with_real(synthetic_texts, real_reviews)
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(diversity_results, bias_results, realism_results)
        
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'num_synthetic_reviews': len(synthetic_reviews),
                'num_real_reviews': len(real_reviews) if real_reviews else 0,
                'provider': synthetic_reviews[0].get('provider', 'unknown') if synthetic_reviews else 'unknown'
            },
            'quality_score': quality_score,
            'diversity': diversity_results,
            'bias': bias_results,
            'realism': realism_results,
            'comparison_with_real': comparison,
            'recommendations': self._generate_recommendations(diversity_results, bias_results, realism_results)
        }
        
        return report
    
    def _compare_with_real(self, synthetic_reviews: List[str], real_reviews: List[str]) -> Dict:
        """
        Compare synthetic reviews with real reviews.
        
        Args:
            synthetic_reviews: List of synthetic review texts
            real_reviews: List of real review texts
            
        Returns:
            Dictionary with comparison metrics
        """
        # Vocabulary comparison
        synthetic_vocab = self.diversity_analyzer.calculate_vocabulary_overlap(synthetic_reviews)
        real_vocab = self.diversity_analyzer.calculate_vocabulary_overlap(real_reviews)
        
        # Length comparison
        synthetic_lengths = [len(r.split()) for r in synthetic_reviews]
        real_lengths = [len(r.split()) for r in real_reviews]
        
        import numpy as np
        
        return {
            'vocabulary': {
                'synthetic_ttr': synthetic_vocab['type_token_ratio'],
                'real_ttr': real_vocab['type_token_ratio'],
                'ttr_difference': round(abs(synthetic_vocab['type_token_ratio'] - real_vocab['type_token_ratio']), 4)
            },
            'length': {
                'synthetic_avg': round(np.mean(synthetic_lengths), 2),
                'real_avg': round(np.mean(real_lengths), 2),
                'length_difference': round(abs(np.mean(synthetic_lengths) - np.mean(real_lengths)), 2)
            },
            'similarity_to_real': self._calculate_cross_similarity(synthetic_reviews, real_reviews)
        }
    
    def _calculate_cross_similarity(self, synthetic_reviews: List[str], real_reviews: List[str]) -> Dict:
        """
        Calculate similarity between synthetic and real reviews.
        
        Args:
            synthetic_reviews: List of synthetic review texts
            real_reviews: List of real review texts
            
        Returns:
            Dictionary with cross-similarity metrics
        """
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        synthetic_embeddings = model.encode(synthetic_reviews)
        real_embeddings = model.encode(real_reviews)
        
        # Calculate cross-similarities
        cross_similarities = cosine_similarity(synthetic_embeddings, real_embeddings)
        
        # For each synthetic review, find max similarity to any real review
        max_similarities = np.max(cross_similarities, axis=1)
        
        return {
            'avg_max_similarity': round(float(np.mean(max_similarities)), 4),
            'min_similarity': round(float(np.min(max_similarities)), 4),
            'max_similarity': round(float(np.max(max_similarities)), 4),
            'interpretation': 'High similarity (>0.7) may indicate copying; very low (<0.3) may indicate unrealistic content'
        }
    
    def _calculate_quality_score(self, diversity: Dict, bias: Dict, realism: Dict) -> Dict:
        """
        Calculate an overall quality score.
        
        Args:
            diversity: Diversity analysis results
            bias: Bias analysis results
            realism: Realism analysis results
            
        Returns:
            Dictionary with quality score and breakdown
        """
        scores = {}
        
        # Diversity score (0-100)
        ttr = diversity['vocabulary']['type_token_ratio']
        avg_sim = diversity['semantic_similarity']['avg_similarity']
        diversity_score = min(100, (ttr * 100 + (1 - avg_sim) * 100) / 2)
        scores['diversity'] = round(diversity_score, 2)
        
        # Bias score (0-100, higher is better = less bias)
        bias_score = 100
        if bias['overall_bias_detected']:
            bias_score -= 30
        if bias['rating_distribution']['is_biased']:
            bias_score -= 20
        if not bias['sentiment_consistency']['is_consistent']:
            bias_score -= 25
        if bias['repetitive_patterns']['has_repetition_issues']:
            bias_score -= 25
        scores['bias'] = max(0, bias_score)
        
        # Realism score (0-100)
        realism_score = 0
        if realism['aspect_coverage'].get('has_good_coverage', False):
            realism_score += 25
        if realism['readability'].get('is_natural', False):
            realism_score += 25
        if realism['ai_patterns'].get('is_realistic', False):
            realism_score += 25
        if realism['pronoun_usage'].get('is_natural', False):
            realism_score += 25
        scores['realism'] = realism_score
        
        # Overall score (weighted average)
        overall = (scores['diversity'] * 0.3 + scores['bias'] * 0.35 + scores['realism'] * 0.35)
        
        return {
            'overall': round(overall, 2),
            'breakdown': scores,
            'grade': self._score_to_grade(overall)
        }
    
    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 90:
            return 'A (Excellent)'
        elif score >= 80:
            return 'B (Good)'
        elif score >= 70:
            return 'C (Acceptable)'
        elif score >= 60:
            return 'D (Poor)'
        else:
            return 'F (Unacceptable)'
    
    def _generate_recommendations(self, diversity: Dict, bias: Dict, realism: Dict) -> List[str]:
        """
        Generate actionable recommendations based on analysis.
        
        Args:
            diversity: Diversity analysis results
            bias: Bias analysis results
            realism: Realism analysis results
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Diversity recommendations
        if diversity['semantic_similarity']['avg_similarity'] > 0.7:
            recommendations.append("Reviews are too similar. Consider increasing temperature or using more diverse personas.")
        
        if diversity['vocabulary']['type_token_ratio'] < 0.3:
            recommendations.append("Limited vocabulary diversity. Encourage more varied word choices in prompts.")
        
        # Bias recommendations
        if bias['rating_distribution']['is_biased']:
            recommendations.append("Rating distribution deviates from expected. Adjust generation parameters or sampling.")
        
        if not bias['sentiment_consistency']['is_consistent']:
            recommendations.append("Sentiment-rating mismatches detected. Review prompt instructions for rating alignment.")
        
        if bias['repetitive_patterns']['has_repetition_issues']:
            recommendations.append("Repetitive phrases detected. Increase generation diversity or use more varied prompts.")
        
        # Realism recommendations
        if not realism['aspect_coverage'].get('has_good_coverage', False):
            recommendations.append("Low product aspect coverage. Emphasize relevant aspects in generation prompts.")
        
        if not realism['readability'].get('is_natural', False):
            recommendations.append("Readability scores outside natural range. Adjust language complexity in prompts.")
        
        if not realism['ai_patterns'].get('is_realistic', False):
            recommendations.append("AI-generated patterns detected. Refine prompts to sound more human and natural.")
        
        if not realism['pronoun_usage'].get('is_natural', False):
            recommendations.append("Low personal pronoun usage. Encourage first-person perspective in prompts.")
        
        if not recommendations:
            recommendations.append("Quality metrics look good! No major issues detected.")
        
        return recommendations
    
    def save_report(self, report: Dict, output_path: str):
        """
        Save report to JSON file.
        
        Args:
            report: Report dictionary
            output_path: Path to save the report
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    
    def print_summary(self, report: Dict):
        """
        Print a human-readable summary of the report.
        
        Args:
            report: Report dictionary
        """
        print("\n" + "="*60)
        print("SYNTHETIC REVIEW QUALITY REPORT")
        print("="*60)
        
        meta = report['metadata']
        print(f"\nGenerated: {meta['generated_at']}")
        print(f"Provider: {meta['provider']}")
        print(f"Synthetic Reviews: {meta['num_synthetic_reviews']}")
        print(f"Real Reviews: {meta['num_real_reviews']}")
        
        print("\n" + "-"*60)
        print("QUALITY SCORE")
        print("-"*60)
        score = report['quality_score']
        print(f"Overall Score: {score['overall']}/100 - {score['grade']}")
        print(f"  - Diversity: {score['breakdown']['diversity']}/100")
        print(f"  - Bias: {score['breakdown']['bias']}/100")
        print(f"  - Realism: {score['breakdown']['realism']}/100")
        
        print("\n" + "-"*60)
        print("RECOMMENDATIONS")
        print("-"*60)
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")
        
        print("\n" + "="*60 + "\n")
