from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, Dict, List
from models.azure_openai_generator import AzureOpenAIGenerator
from models.azure_grok_generator import AzureGrokGenerator
from models.azure_openai_reviewer import AzureOpenAIReviewer
from quality.duplicate_detector import DuplicateDetector
from quality.stratified_sampler import stratified_sample, load_real_reviews
from quality.domain_validator import validate_reviews_batch
from quality.quality_report import QualityReporter
from quality.quality_refiner import QualityRefiner
import json
import yaml
import random
import os
from datetime import datetime
from dotenv import load_dotenv
from utils.fixed_sampler import FixedSampler
# Load environment variables from .env file
load_dotenv()


# ----- 1. Define state -----
class ReviewState(TypedDict):
    persona: dict
    rating: int
    review: Optional[str]
    quality_assessment: Optional[Dict]
    attempt: int
    max_attempts: int
    is_real_review: bool  # Flag to skip quality check for real reviews
    generator_used: Optional[str]  # Track which generator was used
    all_reviews: List[Dict]  # Track all generated reviews for reporting
    attempt_history: List[Dict]  # Track all attempts with their scores


# ----- 2. Load Configuration from YAML -----

def load_config(config_path: str = "config/generation_config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Load configuration
config = load_config()

# Initialize Generators & Reviewer
# Cascading fallback: Grok → Grok with context → Azure OpenAI → Azure OpenAI with more context
grok_generator = AzureGrokGenerator(config=config)
azure_openai_generator = AzureOpenAIGenerator(config=config)

reviewer = AzureOpenAIReviewer(config=config)
quality_reporter = QualityReporter(config=config)


# ----- 3. Node: Generate Review -----
def generate_review_node(state: ReviewState) -> ReviewState:
    """
    Generate a review using cascading fallback strategy.
    Attempt 1: Grok (no context)
    Attempt 2: Real Review (if Grok fails)
    Attempt 3: Azure OpenAI (if real review fails)
    Attempt 4+: Real Review (final fallback)
    """
    attempt_idx = state['attempt']
    max_attempts = state['max_attempts']
    
    print(f"\nAttempt {attempt_idx + 1}/{max_attempts}: ", end="")
    
    real_reviews = []
    current_generator = None
    use_real_review = False
    
    # Cascading Fallback Strategy
    if attempt_idx == 0:
        # Attempt 1: Grok - No context
        try:
            current_generator = grok_generator
            print("Using Grok-4-Fast (No Context)")
        except Exception as e:
            print(f"Grok unavailable ({str(e)}), falling back...")
            current_generator = azure_openai_generator
            
    elif attempt_idx == 1:
        # Attempt 2: Grok with real review examples
        current_generator = grok_generator
        # Load real review examples for context
        try:
            real_reviews_path = "data/real_reviews.jsonl"
            all_reviews = []
            with open(real_reviews_path, 'r', encoding='utf-8') as f:
                for line in f:
                    review_data = json.loads(line.strip())
                    all_reviews.append(review_data['text'])
            if len(all_reviews) >= 3:
                real_reviews = random.sample(all_reviews, 3)
                print("Using Grok-4-Fast (With Context)")
        except Exception:
            real_reviews = []
            print("Using Grok-4-Fast (No Context)")
        
    elif attempt_idx == 2:
        # Attempt 3: Azure OpenAI - With context
        current_generator = azure_openai_generator
        # Load real review examples for context
        try:
            real_reviews_path = "data/real_reviews.jsonl"
            all_reviews = []
            with open(real_reviews_path, 'r', encoding='utf-8') as f:
                for line in f:
                    review_data = json.loads(line.strip())
                    all_reviews.append(review_data['text'])
            if len(all_reviews) >= 3:
                real_reviews = random.sample(all_reviews, 3)
                print("Using Azure OpenAI (With Context)")
        except Exception:
            real_reviews = []
            print("Using Azure OpenAI (No Context)")
            
    else:
        # Attempt 4+: Azure OpenAI with more context
        current_generator = azure_openai_generator
        # Load more real review examples
        try:
            real_reviews_path = "data/real_reviews.jsonl"
            all_reviews = []
            with open(real_reviews_path, 'r', encoding='utf-8') as f:
                for line in f:
                    review_data = json.loads(line.strip())
                    all_reviews.append(review_data['text'])
            if len(all_reviews) >= 5:
                real_reviews = random.sample(all_reviews, 5)
                print("Using Azure OpenAI (With More Context)")
        except Exception:
            real_reviews = []
            print("Using Azure OpenAI (Fallback)")

    # Generate review using current generator (if not using real review)
    if current_generator:
        try:
            review = current_generator.generate_review(
                rating=state['rating'],
                persona=state['persona'],
                real_reviews=real_reviews
            )
            print(f"Generated review: {review[:80]}...")
            state['review'] = review
            state['is_real_review'] = False  # Mark as generated (not real)
            state['generator_used'] = current_generator.get_provider_name()  # Track which generator
        except Exception as e:
            print(f"Generation failed: {str(e)}")
            # Mark as failed so it will retry with next strategy
            state['review'] = None
            state['is_real_review'] = False
            state['generator_used'] = None
    
    state['attempt'] = state['attempt'] + 1
    return state


# ----- 4. Node: Evaluate using Azure OpenAI Reviewer -----
def guardrail_check_node(state: ReviewState) -> ReviewState:
    """Check if the generated review meets quality standards."""
    
    # Skip quality check for real reviews (they're already authentic)
    if state.get('is_real_review', False):
        print("Using real review - skipping quality check")
        state['quality_assessment'] = {
            'pass': True,
            'overall_score': 10.0,
            'issues': []
        }
        return state
    
    print(f"Reviewing generated content...")
    
    try:
        assessment = reviewer.review_generated_content(
            review_text=state["review"],
            rating=state["rating"],
            persona=state["persona"]
        )
        
        print(f"Quality Assessment:")
        print(f"   - Overall Score: {assessment.get('overall_score', 0)}/10")
        print(f"   - Pass: {assessment.get('pass', False)}")
        
        if assessment.get('issues'):
            print(f"   - Issues: {', '.join(assessment['issues'][:3])}")
        
        # Save this attempt to history
        attempt_record = {
            'attempt_number': state['attempt'], # Use current attempt number
            'generator': state.get('generator_used', 'unknown'),
            'review_text': state['review'],
            'quality_score': assessment.get('overall_score', 0),
            'passed': assessment.get('pass', False),
            'issues': assessment.get('issues', [])
        }
        state['attempt_history'].append(attempt_record)
        
        # Track all reviews for final reporting
        state['all_reviews'].append({
            'text': state['review'],
            'quality_score': assessment.get('overall_score', 0)
        })
        
        return {
            **state,
            "quality_assessment": assessment,
            "attempt_history": state['attempt_history'],
            "all_reviews": state['all_reviews']
        }
        
    except Exception as e:
        print(f"Error during review: {str(e)}")
        # If review fails, mark as failed assessment
        failed_assessment = {
            "pass": False,
            "overall_score": 0,
            "issues": [f"Review error: {str(e)}"]
        }
        
        # Save this failed attempt to history
        attempt_history = state.get("attempt_history", [])
        attempt_history.append({
            "attempt_number": state["attempt"],
            "review": state["review"],
            "quality_assessment": failed_assessment,
            "overall_score": 0
        })
        
        return {
            **state,
            "quality_assessment": failed_assessment,
            "attempt_history": attempt_history
        }


# ----- 5. Edge Condition -----
def check_quality_transition(state: ReviewState) -> str:
    """Determine next step based on quality assessment."""
    assessment = state.get("quality_assessment", {})
    passed = assessment.get("pass", False)
    
    if passed:
        print("Review passed quality check!")
        return "good"
    # If max attempts reached, select best attempt
    elif state["attempt"] >= state["max_attempts"]:
        print(f"Max attempts ({state['max_attempts']}) reached.")
        
        attempt_history = state.get("attempt_history", [])
        if attempt_history:
            # Select attempt with highest quality score
            best_attempt = max(attempt_history, key=lambda x: x.get("quality_score", 0))
            print(f"Selecting best review from attempt {best_attempt['attempt_number']} (score: {best_attempt.get('quality_score', 0)}/10)")
            
            # Use the best review
            state["review"] = best_attempt["review_text"]
            state["quality_assessment"] = {
                "overall_score": best_attempt.get("quality_score", 0),
                "pass": True,  # Force pass since we're using best available
                "issues": best_attempt.get("issues", [])
            }
        
        return "give_up"
    else:
        print(f"Quality check failed. Retrying...")
        return "retry"


# ----- 6. Build LangGraph -----
builder = StateGraph(ReviewState)

builder.add_node("generate_review", generate_review_node)
builder.add_node("guardrail_check", guardrail_check_node)

builder.set_entry_point("generate_review")

# Define flow
builder.add_edge("generate_review", "guardrail_check")
builder.add_conditional_edges(
    "guardrail_check",
    check_quality_transition,  # Condition function as positional argument
    {
        "good": END,
        "retry": "generate_review",
        "give_up": END
    }
)

graph = builder.compile()


# ----- 7. Helper Functions -----
def run_review_generation(persona: Dict, rating: int, max_attempts: int = 4) -> Dict:
    """
    Run the review generation workflow for a single review.
    
    Args:
        persona: Persona dictionary
        rating: Rating (0-4)
        max_attempts: Maximum regeneration attempts
        
    Returns:
        Final state dictionary with generated review and assessment
    """
    initial_state: ReviewState = {
        "persona": persona,
        "rating": rating,
        "review": None,
        "quality_assessment": None,
        "attempt": 0,
        "max_attempts": max_attempts,
        "is_real_review": False,
        "generator_used": None,
        "all_reviews": [],
        "attempt_history": []
    }
    
    print(f"\n{'='*60}")
    print(f"Starting review generation for {persona.get('name', 'Unknown')} - Rating: {rating}/4")
    print(f"{'='*60}")
    
    final_state = graph.invoke(initial_state)
    
    print(f"\n{'='*60}")
    print(f"Review generation complete!")
    print(f"Total attempts: {final_state['attempt']}")
    print(f"{'='*60}\n")
    
    return final_state


def generate_batch_reviews(personas: List[Dict], num_reviews: int = 10, max_attempts: int = 4) -> List[Dict]:
    """
    Generate multiple reviews and return them with metadata.
    
    Args:
        personas: List of persona dictionaries
        num_reviews: Number of reviews to generate
        max_attempts: Maximum regeneration attempts per review
        
    Returns:
        List of review dictionaries with metadata
    """
    
    # Load rating distribution from config
    rating_distribution = config.get('rating_distribution')
    
    # Initialize fixed sampler for deterministic rating distribution
    fixed_sampler = FixedSampler()
    
    reviews = []
    
    for i in range(num_reviews):
        # Select random persona but use fixed sampling for rating
        persona = random.choice(personas)
        rating = fixed_sampler.get_next_rating(rating_distribution, num_reviews)
        
        print(f"\n{'#'*60}")
        print(f"Review {i+1}/{num_reviews}")
        print(f"{'#'*60}")
        
        # Run generation workflow
        result = run_review_generation(persona, rating, max_attempts)
        
        # Use the tracked generator name from state
        model_used = result.get('generator_used')
        
        # Fallback: if generator_used wasn't set, infer from attempt number
        if not model_used or model_used == 'unknown':
            final_attempt = result.get('attempt', 1)
            if final_attempt <= 2:
                model_used = grok_generator.get_provider_name()
            else:
                model_used = azure_openai_generator.get_provider_name()
            print(f"Generator not tracked, inferred from attempt {final_attempt}: {model_used}")
        else:
            print(f"Generator tracked: {model_used}")
        
        # Store review with metadata
        review_data = {
            "text": result["review"],
            "rating": result["rating"],
            "persona": result["persona"],
            "model": model_used,  # Dynamic model name
            "attempts": result["attempt"],
            "quality_assessment": result["quality_assessment"],
            "attempt_history": result.get("attempt_history", []),  # All attempts with details
            "generated_at": datetime.now().isoformat()
        }
        
        reviews.append(review_data)
    
    print(f"\n{'='*60}")
    print(f"Checking for Semantic Duplicates")
    print(f"{'='*60}\n")
    
    # Detect and remove duplicates using FAISS
    
    detector = DuplicateDetector(similarity_threshold=0.95)
    unique_reviews, dup_stats = detector.remove_duplicates(reviews)
    
    print(f"Duplicate Detection Results:")
    print(f"   - Original count: {dup_stats['original_count']}")
    print(f"   - Duplicates found: {dup_stats['duplicate_count']}")
    print(f"   - Unique reviews: {dup_stats['unique_count']}")
    print(f"   - Duplicate rate: {dup_stats['duplicate_rate']*100:.1f}%")
    
    # Regenerate reviews if needed to reach target count
    if len(unique_reviews) < num_reviews:
        needed = num_reviews - len(unique_reviews)
        print(f"\nRegenerating {needed} reviews to reach target count...")
        
        iteration = 1
        while len(unique_reviews) < num_reviews and iteration <= 5:  # Max 5 iterations
            print(f"\n   Iteration {iteration}: Generating {needed} additional reviews...")
            
            for i in range(needed):
                persona = random.choice(personas)
                rating = fixed_sampler.get_next_rating(rating_distribution, num_reviews) # Use the sampler instance
                
                result = run_review_generation(persona, rating, max_attempts)
                
                review_data = {
                    "text": result["review"],
                    "rating": result["rating"],
                    "persona": result["persona"],
                    "provider": "local_ensemble",
                    "attempts": result["attempt"],
                    "quality_assessment": result["quality_assessment"],
                    "generated_at": datetime.now().isoformat()
                }
                
                unique_reviews.append(review_data)
            
            # Check for duplicates again
            unique_reviews, dup_stats = detector.remove_duplicates(unique_reviews)
            needed = num_reviews - len(unique_reviews)
            iteration += 1
        
        print(f"\nFinal count after deduplication: {len(unique_reviews)} reviews")
    
    reviews = unique_reviews
    
    return reviews


def generate_quality_report(reviews: List[Dict], output_dir: str = "reports"):
    """
    Generate a comprehensive quality report for the generated reviews.
    
    Args:
        reviews: List of review dictionaries
        output_dir: Base directory for reports (default: "reports")
        
    Returns:
        Quality report dictionary
    """
    print(f"\n{'='*60}")
    print("Generating Quality Report")
    print(f"{'='*60}\n")
    
    # Create timestamped folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = f"{output_dir}/report_{timestamp}"
    os.makedirs(report_dir, exist_ok=True)
    
    # Sample real reviews preserving ORIGINAL distribution
    
    try:
        all_real_reviews = load_real_reviews("data/real_reviews.jsonl")
        sampled_real_reviews = stratified_sample(all_real_reviews, sample_size=len(reviews))
        print(f"Sampled {len(sampled_real_reviews)} real reviews with original distribution")
    except Exception as e:
        print(f"Could not load real reviews: {str(e)}")
        sampled_real_reviews = []
    
    # Validate domain (shoe-related content)
    domain_validation = validate_reviews_batch(reviews)
    print(f"Domain validation: {domain_validation['shoe_related_percentage']:.1f}% shoe-related")
    
    # Generate report with comparisons
    report = quality_reporter.generate_report(reviews, sampled_real_reviews)
    
    # Add domain validation to report
    report['domain_validation'] = domain_validation
    
    # Save generated reviews to the report folder in JSONL format
    reviews_path = f"{report_dir}/generated_reviews.jsonl"
    with open(reviews_path, 'w', encoding='utf-8') as f:
        for review in reviews:
            # Convert to format matching real reviews: labels, text, persona, model
            review_dict = {
                "labels": review["rating"],
                "text": review["text"],
                "persona": review["persona"]["name"],
                "model": review.get("model", "unknown")  # Use tracked model name
            }
            f.write(json.dumps(review_dict, ensure_ascii=False) + '\n')
    print(f"Generated reviews saved to: {reviews_path}")
    
    # Save detailed attempt history to separate file
    attempt_history_path = f"{report_dir}/attempt_history.jsonl"
    with open(attempt_history_path, 'w', encoding='utf-8') as f:
        for i, review in enumerate(reviews):
            attempt_log = {
                "review_index": i + 1,
                "persona": review.get('persona', {}).get('name', 'unknown'),
                "rating": review.get('rating', 2),
                "final_model": review.get('model', 'unknown'),
                "total_attempts": review.get('attempts', 1),
                "final_quality_score": review.get('quality_assessment', {}).get('overall_score', 0),
                "attempt_history": review.get('attempt_history', [])
            }
            f.write(json.dumps(attempt_log, ensure_ascii=False) + '\n')
    print(f"Attempt history saved to: {attempt_history_path}")
    
    # Save reports (both JSON and Markdown)
    report_path_json = f"{report_dir}/quality_report.json"
    report_path_md = f"{report_dir}/quality_report.md"
    
    quality_reporter.save_report(report, report_path_json)
    quality_reporter.save_report_markdown(report, report_path_md)
    
    print(f"Report saved to: {report_path_md}")
    print(f"JSON report saved to: {report_path_json}")
    
    # Iterative Quality Refinement Loop (3 attempts)
    print(f"\n{'='*60}")
    print(f"Quality Refinement Loop")
    print(f"{'='*60}\n")
    
    # Load config for refinement
    personas = config.get('personas')
    rating_distribution = config.get('rating_distribution')
    
    refiner = QualityRefiner()
    best_reviews = reviews
    best_report = report
    best_score = report['quality_score']['overall']
    
    print(f"Initial Quality Score: {best_score:.1f}/100")
    
    # Get configuration for refinement
    personas = config.get('personas')
    rating_distribution = config.get('rating_distribution')

    for attempt in range(1, 4):  # 3 refinement attempts
        print(f"\nRefinement Attempt {attempt}/3:")
        
        # Check if quality is already good enough
        if best_score >= 70:
            print(f"   Quality score {best_score:.1f}/100 is acceptable. Skipping refinement.")
            break
        
        # Identify and fix problems
        refined_reviews, refine_stats, removed_count = refiner.refine_reviews(
            reviews=best_reviews,
            quality_report=best_report,
            generator=grok_generator,
            reviewer=reviewer,
            personas=personas,
            rating_distribution=rating_distribution,
            real_reviews=sampled_real_reviews,
            max_attempts=3
        )
        
        if removed_count == 0:
            print(f"   No problematic reviews found. Quality is optimal for current settings.")
            break
        
        print(f"   - Removed: {refine_stats['removed']} problematic reviews")
        print(f"   - Regenerated: {refine_stats['regenerated']} new reviews")
        
        # Generate new report for refined reviews
        print(f"   - Generating new quality report...")
        new_report = quality_reporter.generate_report(refined_reviews, sampled_real_reviews)
        new_report['domain_validation'] = domain_validation
        new_score = new_report['quality_score']['overall']
        
        print(f"   New Quality Score: {new_score:.1f}/100 (was {best_score:.1f}/100)")
        
        # Keep the best version
        if new_score > best_score:
            print(f"   Improvement! Keeping refined version (+{new_score - best_score:.1f} points)")
            best_reviews = refined_reviews
            best_report = new_report
            best_score = new_score
        else:
            print(f"   No improvement. Keeping previous version.")
            break  # Stop if no improvement
    
    # Use the best version
    reviews = best_reviews
    report = best_report
    
    print(f"\nFinal Quality Score: {best_score:.1f}/100")
    
    # Save final reports
    quality_reporter.save_report(report, report_path_json)
    quality_reporter.save_report_markdown(report, report_path_md)
    
    print(f"Final report saved to: {report_path_md}")
    print(f"Final JSON report saved to: {report_path_json}")
    
    # Generate comparison plots
    print(f"Generating comparison plots...")
    if sampled_real_reviews and len(sampled_real_reviews) > 0:
        quality_reporter.generate_comparison_plots(reviews, sampled_real_reviews, report_dir)
        print(f"Comparison plots saved to: {report_dir}/")
    else:
        print(f"Skipping comparison plots (no real reviews available)")
    
    # Generate distribution analysis
    print(f"Generating distribution analysis...")
    quality_reporter.generate_distribution_plots(reviews, report_dir)
    print(f"Distribution analysis saved to: {report_dir}/distribution_analysis.png")
    
    # Print summary
    quality_reporter.print_summary(report)
    
    return report
