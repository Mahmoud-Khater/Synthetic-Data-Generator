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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
import os

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
    
    def _compare_with_real(self, synthetic_reviews: List[str], real_reviews: List) -> Dict:
        """
        Compare synthetic reviews with real reviews.
        
        Args:
            synthetic_reviews: List of synthetic review texts
            real_reviews: List of real reviews (can be strings or dicts with 'text' field)
            
        Returns:
            Dictionary with comparison metrics
        """
        # Extract text from real reviews if they are dictionaries
        if real_reviews and isinstance(real_reviews[0], dict):
            real_review_texts = [r.get('text', '') for r in real_reviews]
        else:
            real_review_texts = real_reviews
        
        # Vocabulary comparison
        synthetic_vocab = self.diversity_analyzer.calculate_vocabulary_overlap(synthetic_reviews)
        real_vocab = self.diversity_analyzer.calculate_vocabulary_overlap(real_review_texts)
        
        # Length comparison
        synthetic_lengths = [len(r.split()) for r in synthetic_reviews]
        real_lengths = [len(r.split()) for r in real_review_texts]
        
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
    
    def _calculate_cross_similarity(self, synthetic_reviews: List[str], real_reviews: List) -> Dict:
        """
        Calculate similarity between synthetic and real reviews.
        
        Args:
            synthetic_reviews: List of synthetic review texts
            real_reviews: List of real reviews (can be strings or dicts with 'text' field)
            
        Returns:
            Dictionary with cross-similarity metrics
        """
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        # Extract text from real reviews if they are dictionaries
        if real_reviews and isinstance(real_reviews[0], dict):
            real_review_texts = [r.get('text', '') for r in real_reviews]
        else:
            real_review_texts = real_reviews
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        synthetic_embeddings = model.encode(synthetic_reviews)
        real_embeddings = model.encode(real_review_texts)
        
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
        # Custom JSON encoder to handle numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.bool_):
                    return bool(obj)
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    def save_report_markdown(self, report: Dict, output_path: str):
        """
        Save report to Markdown file.
        
        Args:
            report: Report dictionary
            output_path: Path to save the markdown report
        """
        md_content = f"""# Synthetic Review Quality Report

## Metadata
- **Generated**: {report['metadata']['generated_at']}
- **Provider**: {report['metadata']['provider']}
- **Synthetic Reviews**: {report['metadata']['num_synthetic_reviews']}
- **Real Reviews**: {report['metadata']['num_real_reviews']}

---

## Quality Score

**Overall**: {report['quality_score']['overall']:.1f}/100 - **{report['quality_score']['grade']}**

| Metric | Score |
|--------|-------|
| Diversity | {report['quality_score']['breakdown']['diversity']:.2f}/100 |
| Bias | {report['quality_score']['breakdown']['bias']:.2f}/100 |
| Realism | {report['quality_score']['breakdown']['realism']:.2f}/100 |

---

## Diversity Analysis

### Vocabulary
- **Type-Token Ratio**: {report['diversity']['vocabulary']['type_token_ratio']:.3f}
- **Unique Tokens**: {report['diversity']['vocabulary']['unique_tokens']}
- **Total Tokens**: {report['diversity']['vocabulary']['total_tokens']}
- **Avg Unique per Review**: {report['diversity']['vocabulary']['avg_unique_per_review']:.2f}

### Semantic Similarity
- **Average**: {report['diversity']['semantic_similarity']['avg_similarity']:.3f}
- **Max**: {report['diversity']['semantic_similarity']['max_similarity']:.3f}
- **Min**: {report['diversity']['semantic_similarity']['min_similarity']:.3f}
- **Std Dev**: {report['diversity']['semantic_similarity']['std_similarity']:.4f}

---

## Bias Detection

### Rating Distribution
| Rating | Actual | Expected | Deviation |
|--------|--------|----------|-----------|
"""
        # Add rating distribution table - use 0-4 rating scale
        for rating in range(5):  # 0, 1, 2, 3, 4
            actual = report['bias']['rating_distribution']['actual_distribution'].get(str(rating), 0)
            expected = report['bias']['rating_distribution']['expected_distribution'].get(str(rating), 0)
            deviation = report['bias']['rating_distribution']['deviations'].get(str(rating), 0)
            md_content += f"| {rating} | {actual*100:.1f}% | {expected*100:.1f}% | {abs(deviation)*100:.1f}% |\n"
        
        md_content += f"""
**Total Deviation**: {report['bias']['rating_distribution']['total_deviation']*100:.1f}%  
**Biased**: {'Yes' if report['bias']['rating_distribution']['is_biased'] else 'No'}

### Sentiment Consistency
- **Inconsistencies**: {report['bias']['sentiment_consistency']['inconsistencies_found']}/{report['bias']['sentiment_consistency']['total_reviews']}
- **Inconsistency Rate**: {report['bias']['sentiment_consistency']['inconsistency_rate']*100:.1f}%
- **Consistent**: {'Yes' if report['bias']['sentiment_consistency']['is_consistent'] else 'No'}

### Repetitive Patterns
- **Unique Reviews**: {report['bias']['repetitive_patterns']['unique_reviews']}/{report['bias']['repetitive_patterns']['total_reviews']}
- **Duplicate Rate**: {report['bias']['repetitive_patterns']['duplicate_rate']*100:.1f}%
- **Has Repetition Issues**: {'Yes' if report['bias']['repetitive_patterns']['has_repetition_issues'] else 'No'}

**Top Repetitive Phrases**:
"""
        for phrase in report['bias']['repetitive_patterns']['repetitive_phrases'][:10]:
            md_content += f"- \"{phrase['phrase']}\" - {phrase['count']} times ({phrase['frequency']*100:.1f}%)\n"
        
        md_content += f"""
---

## Realism Analysis

### Aspect Coverage
- **Coverage Rate**: {report['realism']['aspect_coverage']['coverage_rate']*100:.1f}%
- **Avg Mentions per Aspect**: {report['realism']['aspect_coverage']['avg_mentions_per_aspect']:.2f}
- **Has Good Coverage**: {'Yes' if report['realism']['aspect_coverage']['has_good_coverage'] else 'No'}

**Aspect Mentions**:
"""
        for aspect, count in report['realism']['aspect_coverage']['aspect_mentions'].items():
            md_content += f"- **{aspect}**: {count}\n"
        
        md_content += f"""
### Readability
- **Avg Flesch Score**: {report['realism']['readability']['avg_flesch_score']:.2f}
- **Interpretation**: {report['realism']['readability']['interpretation']}
- **Is Natural**: {'Yes' if report['realism']['readability']['is_natural'] else 'No'}

### AI Patterns
- **AI Pattern Rate**: {report['realism']['ai_patterns']['ai_pattern_rate']*100:.1f}%
- **Is Realistic**: {'Yes' if report['realism']['ai_patterns']['is_realistic'] else 'No'}

---

## Comparison with Real Reviews

| Metric | Synthetic | Real | Difference |
|--------|-----------|------|------------|
| Vocabulary (TTR) | {report['comparison_with_real']['vocabulary']['synthetic_ttr']:.3f} | {report['comparison_with_real']['vocabulary']['real_ttr']:.3f} | {report['comparison_with_real']['vocabulary']['ttr_difference']:.3f} |
| Avg Length (words) | {report['comparison_with_real']['length']['synthetic_avg']:.1f} | {report['comparison_with_real']['length']['real_avg']:.1f} | {report['comparison_with_real']['length']['length_difference']:.1f} |
| Similarity to Real | {report['comparison_with_real']['similarity_to_real']['avg_max_similarity']:.3f} | - | - |

---

## Domain Validation

- **Shoe-Related**: {report['domain_validation']['shoe_related_count']}/{report['domain_validation']['total_reviews']} ({report['domain_validation']['shoe_related_percentage']:.1f}%)
- **Avg Relevance Score**: {report['domain_validation']['avg_relevance_score']:.3f}
- **Avg Keyword Count**: {report['domain_validation']['avg_keyword_count']:.1f}
- **Flagged Reviews**: {report['domain_validation']['flagged_count']}

---

## Recommendations

"""
        for i, rec in enumerate(report['recommendations'], 1):
            md_content += f"{i}. {rec}\n"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
    
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
    
    def generate_comparison_plots(self, synthetic_reviews: List[Dict], real_reviews: List[Dict], output_dir: str):
        """
        Generate all comparison plots between synthetic and real reviews.
        
        Args:
            synthetic_reviews: List of synthetic review dictionaries
            real_reviews: List of real review dictionaries
            output_dir: Directory to save plots
        """
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        
        # Generate each plot
        self._plot_sentiment_comparison(synthetic_reviews, real_reviews, output_dir)
        self._plot_length_distribution(synthetic_reviews, real_reviews, output_dir)
        self._plot_rating_distribution(synthetic_reviews, real_reviews, output_dir)
        self._plot_semantic_similarity(synthetic_reviews, real_reviews, output_dir)
    
    def _plot_sentiment_comparison(self, synthetic_reviews: List[Dict], real_reviews: List[Dict], output_dir: str):
        """Plot sentiment distribution comparison."""
        
        def get_sentiment_from_rating(rating):
            """Map rating (0-4) to sentiment category."""
            if rating >= 3:  # 3-4
                return 'Positive'
            elif rating == 2:
                return 'Neutral'
            else:  # 0-1
                return 'Negative'
        
        # Get sentiments from ratings
        synthetic_sentiments = [get_sentiment_from_rating(r.get('rating', 2)) for r in synthetic_reviews]
        real_sentiments = [get_sentiment_from_rating(r.get('labels', r.get('rating', 2))) for r in real_reviews]
        
        # Count sentiments
        synthetic_counts = Counter(synthetic_sentiments)
        real_counts = Counter(real_sentiments)
        
        # Calculate percentages
        categories = ['Positive', 'Neutral', 'Negative']
        synthetic_pcts = [synthetic_counts.get(c, 0) / len(synthetic_reviews) * 100 for c in categories]
        real_pcts = [real_counts.get(c, 0) / len(real_reviews) * 100 for c in categories]
        
        # Plot
        fig, ax = plt.subplots()
        x = np.arange(len(categories))
        width = 0.35
        
        ax.bar(x - width/2, synthetic_pcts, width, label='Synthetic', color='#3498db')
        ax.bar(x + width/2, real_pcts, width, label='Real', color='#2ecc71')
        
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Sentiment Distribution Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/sentiment_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_length_distribution(self, synthetic_reviews: List[Dict], real_reviews: List[Dict], output_dir: str):
        """Plot review length distribution comparison."""
        synthetic_lengths = [len(r['text'].split()) for r in synthetic_reviews]
        real_lengths = [len(r.get('text', '').split()) for r in real_reviews]
        
        fig, ax = plt.subplots()
        
        # Create histograms
        bins = np.linspace(0, max(max(synthetic_lengths), max(real_lengths)), 20)
        ax.hist(synthetic_lengths, bins=bins, alpha=0.6, label='Synthetic', color='#3498db', edgecolor='black')
        ax.hist(real_lengths, bins=bins, alpha=0.6, label='Real', color='#2ecc71', edgecolor='black')
        
        ax.set_xlabel('Review Length (words)')
        ax.set_ylabel('Frequency')
        ax.set_title('Review Length Distribution')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add mean lines
        ax.axvline(np.mean(synthetic_lengths), color='#3498db', linestyle='--', linewidth=2, label=f'Synthetic Mean: {np.mean(synthetic_lengths):.1f}')
        ax.axvline(np.mean(real_lengths), color='#2ecc71', linestyle='--', linewidth=2, label=f'Real Mean: {np.mean(real_lengths):.1f}')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/length_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_rating_distribution(self, synthetic_reviews: List[Dict], real_reviews: List[Dict], output_dir: str):
        """Plot rating distribution comparison."""
        synthetic_ratings = [r.get('rating', 3) for r in synthetic_reviews]
        real_ratings = [r.get('labels', r.get('rating', 3)) for r in real_reviews]
        
        # Count ratings
        synthetic_counts = Counter(synthetic_ratings)
        real_counts = Counter(real_ratings)
        
        # Calculate percentages
        ratings = [0, 1, 2, 3, 4]
        synthetic_pcts = [synthetic_counts.get(r, 0) / len(synthetic_reviews) * 100 for r in ratings]
        real_pcts = [real_counts.get(r, 0) / len(real_reviews) * 100 for r in ratings]
        
        # Plot
        fig, ax = plt.subplots()
        x = np.arange(len(ratings))
        width = 0.35
        
        ax.bar(x - width/2, synthetic_pcts, width, label='Synthetic', color='#3498db')
        ax.bar(x + width/2, real_pcts, width, label='Real', color='#2ecc71')
        
        ax.set_xlabel('Rating')
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Rating Distribution Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(ratings)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/rating_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_semantic_similarity(self, synthetic_reviews: List[Dict], real_reviews: List[Dict], output_dir: str):
        """Plot semantic similarity distribution."""
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Load model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get embeddings
        synthetic_texts = [r['text'] for r in synthetic_reviews]
        real_texts = [r.get('text', '') for r in real_reviews]
        
        synthetic_embeddings = model.encode(synthetic_texts)
        real_embeddings = model.encode(real_texts)
        
        # Calculate pairwise similarities
        similarities = []
        for syn_emb in synthetic_embeddings:
            # Find max similarity to any real review
            sims = cosine_similarity([syn_emb], real_embeddings)[0]
            similarities.append(max(sims))
        
        # Plot
        fig, ax = plt.subplots()
        ax.hist(similarities, bins=20, color='#9b59b6', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Cosine Similarity to Closest Real Review')
        ax.set_ylabel('Frequency')
        ax.set_title('Semantic Similarity Distribution')
        ax.axvline(np.mean(similarities), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(similarities):.3f}')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/semantic_similarity.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_distribution_plots(self, reviews: List[Dict], output_dir: str):
        """
        Generate distribution analysis plots for generated reviews.
        
        Args:
            reviews: List of review dictionaries
            output_dir: Directory to save plots
        """
        # Set style
        sns.set_style("whitegrid")
        
        # Extract data
        quality_scores = [r.get('quality_assessment', {}).get('overall_score', 0) for r in reviews]
        word_counts = [len(r['text'].split()) for r in reviews]
        ratings = [r.get('rating', 3) for r in reviews]
        
        # Calculate sentiment scores
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = [analyzer.polarity_scores(r['text'])['compound'] for r in reviews]
        
        # Create 2x2 subplot grid
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Quality Score Distribution
        ax1.hist(quality_scores, bins=15, color='#3498db', edgecolor='black', alpha=0.7)
        ax1.axvline(np.mean(quality_scores), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(quality_scores):.2f}')
        ax1.axvline(np.median(quality_scores), color='green', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(quality_scores):.2f}')
        ax1.set_xlabel('Quality Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Quality Score Distribution')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Word Count Distribution
        ax2.hist(word_counts, bins=15, color='#2ecc71', edgecolor='black', alpha=0.7)
        ax2.axvline(np.mean(word_counts), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(word_counts):.1f}')
        ax2.axvline(np.median(word_counts), color='green', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(word_counts):.1f}')
        ax2.set_xlabel('Word Count')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Review Length Distribution')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Sentiment Score Distribution
        ax3.hist(sentiment_scores, bins=15, color='#9b59b6', edgecolor='black', alpha=0.7)
        ax3.axvline(np.mean(sentiment_scores), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(sentiment_scores):.3f}')
        ax3.axvline(np.median(sentiment_scores), color='green', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(sentiment_scores):.3f}')
        ax3.set_xlabel('Sentiment Score (-1 to 1)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Sentiment Distribution')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Rating Distribution
        rating_counts = Counter(ratings)
        ratings_list = sorted(rating_counts.keys())
        counts = [rating_counts[r] for r in ratings_list]
        ax4.bar(ratings_list, counts, color='#e74c3c', edgecolor='black', alpha=0.7)
        ax4.set_xlabel('Rating (0-4 stars)')
        ax4.set_ylabel('Count')
        ax4.set_title('Rating Distribution')
        ax4.set_xticks(ratings_list)
        ax4.grid(axis='y', alpha=0.3)
        
        # Add statistics text
        stats_text = f'Total Reviews: {len(reviews)}\n'
        stats_text += f'Quality: μ={np.mean(quality_scores):.2f}, σ={np.std(quality_scores):.2f}\n'
        stats_text += f'Length: μ={np.mean(word_counts):.1f}, σ={np.std(word_counts):.1f}\n'
        stats_text += f'Sentiment: μ={np.mean(sentiment_scores):.3f}, σ={np.std(sentiment_scores):.3f}'
        
        fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(f"{output_dir}/distribution_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
