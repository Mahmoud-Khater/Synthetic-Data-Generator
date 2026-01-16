from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, Dict, List
from models.azure_openai_generator import AzureOpenAIGenerator
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

# Initialize Azure OpenAI Generator & Reviewer
generator = AzureOpenAIGenerator(config=config)
reviewer = AzureOpenAIReviewer(config=config)
quality_reporter = QualityReporter(config=config)


# ----- 3. Node: Generate Review -----
def generate_review_node(state: ReviewState) -> ReviewState:
    """Generate a review using Azure OpenAI."""
    print(f"\n🔄 Attempt {state['attempt'] + 1}/{state['max_attempts']}: Generating review...")
    
    # Load real reviews for context if this is a retry (attempt > 0)
    real_reviews = []
    if state["attempt"] > 0:
        try:
            real_reviews_path = "data/real_reviews.jsonl"
            
            # Read all reviews from JSONL file
            all_reviews = []
            with open(real_reviews_path, 'r', encoding='utf-8') as f:
                for line in f:
                    review_data = json.loads(line.strip())
                    all_reviews.append(review_data['text'])
            
            # Select 3 random reviews
            if len(all_reviews) >= 3:
                real_reviews = random.sample(all_reviews, 3)
                print(f"📚 Loaded 3 real review examples for context")
            
        except Exception as e:
            print(f"⚠️  Could not load real reviews: {str(e)}")
            real_reviews = []
    
    review = generator.generate_review(
        rating=state["rating"],
        persona=state["persona"],
        real_reviews=real_reviews
    )
    
    print(f"✅ Generated review: {review[:100]}...")
    
    return {
        **state,
        "review": review,
        "attempt": state["attempt"] + 1
    }


# ----- 4. Node: Evaluate using Azure OpenAI Reviewer -----
def guardrail_check_node(state: ReviewState) -> ReviewState:
    """Review the generated content using Azure OpenAI reviewer."""
    print(f"🔍 Reviewing generated content...")
    
    try:
        assessment = reviewer.review_generated_content(
            review_text=state["review"],
            rating=state["rating"],
            persona=state["persona"]
        )
        
        print(f"📊 Quality Assessment:")
        print(f"   - Overall Score: {assessment.get('overall_score', 0)}/10")
        print(f"   - Pass: {assessment.get('pass', False)}")
        
        if assessment.get('issues'):
            print(f"   - Issues: {', '.join(assessment['issues'][:3])}")
        
        # Save this attempt to history
        attempt_history = state.get("attempt_history", [])
        attempt_history.append({
            "attempt_number": state["attempt"],
            "review": state["review"],
            "quality_assessment": assessment,
            "overall_score": assessment.get("overall_score", 0)
        })
        
        return {
            **state,
            "quality_assessment": assessment,
            "attempt_history": attempt_history
        }
        
    except Exception as e:
        print(f"⚠️  Error during review: {str(e)}")
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
        print("✅ Review passed quality check!")
        return "good"
    elif state["attempt"] >= state["max_attempts"]:
        print(f"⚠️  Max attempts ({state['max_attempts']}) reached.")
        
        # Select the best review from all attempts
        attempt_history = state.get("attempt_history", [])
        if attempt_history:
            # Find the attempt with the highest score
            best_attempt = max(attempt_history, key=lambda x: x["overall_score"])
            best_score = best_attempt["overall_score"]
            best_attempt_num = best_attempt["attempt_number"]
            
            print(f"📊 Selecting best review from attempt {best_attempt_num} (score: {best_score}/10)")
            
            # Update state with the best review
            state["review"] = best_attempt["review"]
            state["quality_assessment"] = best_attempt["quality_assessment"]
        
        return "give_up"
    else:
        print(f"🔄 Quality check failed. Retrying...")
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
def run_review_generation(persona: Dict, rating: int, max_attempts: int = 3) -> Dict:
    """
    Run the review generation workflow for a single review.
    
    Args:
        persona: Persona dictionary
        rating: Rating (0-4)
        max_attempts: Maximum regeneration attempts
        
    Returns:
        Final state dictionary with generated review and assessment
    """
    initial_state = {
        "persona": persona,
        "rating": rating,
        "review": None,
        "quality_assessment": None,
        "attempt": 0,
        "max_attempts": max_attempts,
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


def generate_batch_reviews(personas: List[Dict], num_reviews: int = 10, max_attempts: int = 3) -> List[Dict]:
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
    config = load_config()
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
        
        # Store review with metadata
        review_data = {
            "text": result["review"],
            "rating": result["rating"],
            "persona": result["persona"],
            "provider": generator.get_provider_name(),
            "attempts": result["attempt"],
            "quality_assessment": result["quality_assessment"],
            "generated_at": datetime.now().isoformat()
        }
        
        reviews.append(review_data)
    
    print(f"\n{'='*60}")
    print(f"Checking for Semantic Duplicates")
    print(f"{'='*60}\n")
    
    # Detect and remove duplicates using FAISS
    
    detector = DuplicateDetector(similarity_threshold=0.95)
    unique_reviews, dup_stats = detector.remove_duplicates(reviews)
    
    print(f"📊 Duplicate Detection Results:")
    print(f"   - Original count: {dup_stats['original_count']}")
    print(f"   - Duplicates found: {dup_stats['duplicate_count']}")
    print(f"   - Unique reviews: {dup_stats['unique_count']}")
    print(f"   - Duplicate rate: {dup_stats['duplicate_rate']*100:.1f}%")
    
    # Regenerate reviews if needed to reach target count
    if len(unique_reviews) < num_reviews:
        needed = num_reviews - len(unique_reviews)
        print(f"\n🔄 Regenerating {needed} reviews to reach target count...")
        
        iteration = 1
        while len(unique_reviews) < num_reviews and iteration <= 5:  # Max 5 iterations
            print(f"\n   Iteration {iteration}: Generating {needed} additional reviews...")
            
            for i in range(needed):
                persona = random.choice(personas)
                rating = select_rating(rating_distribution)
                
                result = run_review_generation(persona, rating, max_attempts)
                
                review_data = {
                    "text": result["review"],
                    "rating": result["rating"],
                    "persona": result["persona"],
                    "provider": generator.get_provider_name(),
                    "attempts": result["attempt"],
                    "quality_assessment": result["quality_assessment"],
                    "generated_at": datetime.now().isoformat()
                }
                
                unique_reviews.append(review_data)
            
            # Check for duplicates again
            unique_reviews, dup_stats = detector.remove_duplicates(unique_reviews)
            needed = num_reviews - len(unique_reviews)
            iteration += 1
        
        print(f"\n✅ Final count after deduplication: {len(unique_reviews)} reviews")
    
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
        print(f"✅ Sampled {len(sampled_real_reviews)} real reviews with original distribution")
    except Exception as e:
        print(f"⚠️  Could not load real reviews: {str(e)}")
        sampled_real_reviews = []
    
    # Validate domain (shoe-related content)
    domain_validation = validate_reviews_batch(reviews)
    print(f"✅ Domain validation: {domain_validation['shoe_related_percentage']:.1f}% shoe-related")
    
    # Generate report with comparisons
    report = quality_reporter.generate_report(reviews, sampled_real_reviews)
    
    # Add domain validation to report
    report['domain_validation'] = domain_validation
    
    # Save generated reviews to the report folder in JSONL format
    reviews_path = f"{report_dir}/generated_reviews.jsonl"
    with open(reviews_path, 'w', encoding='utf-8') as f:
        for review in reviews:
            # Convert to format matching real reviews: labels, text, persona, model
            jsonl_entry = {
                "labels": review.get('rating', 2),  # Rating 0-4
                "text": review.get('text', ''),
                "persona": review.get('persona', {}).get('name', 'unknown'),
                "model": "azure_openai_gpt-4.1-mini"
            }
            f.write(json.dumps(jsonl_entry, ensure_ascii=False) + '\n')
    print(f"💾 Generated reviews saved to: {reviews_path}")
    
    # Save reports (both JSON and Markdown)
    report_path_json = f"{report_dir}/quality_report.json"
    report_path_md = f"{report_dir}/quality_report.md"
    
    quality_reporter.save_report(report, report_path_json)
    quality_reporter.save_report_markdown(report, report_path_md)
    
    print(f"📄 Report saved to: {report_path_md}")
    print(f"📄 JSON report saved to: {report_path_json}")
    
    # Iterative Quality Refinement Loop (3 attempts)
    print(f"\n{'='*60}")
    print(f"Quality Refinement Loop")
    print(f"{'='*60}\n")
    
    
    refiner = QualityRefiner()
    best_reviews = reviews
    best_report = report
    best_score = report['quality_score']['overall']
    
    print(f"📊 Initial Quality Score: {best_score:.1f}/100")
    
    # Get configuration for refinement
    personas = config.get('personas')
    rating_distribution = config.get('rating_distribution')

    for attempt in range(1, 4):  # 3 refinement attempts
        print(f"\n🔄 Refinement Attempt {attempt}/3:")
        
        # Check if quality is already good enough
        if best_score >= 70:
            print(f"   ✅ Quality score {best_score:.1f}/100 is acceptable. Skipping refinement.")
            break
        
        # Identify and fix problems
        refined_reviews, refine_stats, removed_count = refiner.refine_reviews(
            reviews=best_reviews,
            quality_report=best_report,
            generator=generator,
            reviewer=reviewer,
            personas=personas,
            rating_distribution=rating_distribution,
            real_reviews=sampled_real_reviews,
            max_attempts=3
        )
        
        if removed_count == 0:
            print(f"   ℹ️  No problematic reviews found. Quality is optimal for current settings.")
            break
        
        print(f"   - Removed: {refine_stats['removed']} problematic reviews")
        print(f"   - Regenerated: {refine_stats['regenerated']} new reviews")
        
        # Generate new report for refined reviews
        print(f"   - Generating new quality report...")
        new_report = quality_reporter.generate_report(refined_reviews, sampled_real_reviews)
        new_report['domain_validation'] = domain_validation
        new_score = new_report['quality_score']['overall']
        
        print(f"   📊 New Quality Score: {new_score:.1f}/100 (was {best_score:.1f}/100)")
        
        # Keep the best version
        if new_score > best_score:
            print(f"   ✅ Improvement! Keeping refined version (+{new_score - best_score:.1f} points)")
            best_reviews = refined_reviews
            best_report = new_report
            best_score = new_score
        else:
            print(f"   ⚠️  No improvement. Keeping previous version.")
            break  # Stop if no improvement
    
    # Use the best version
    reviews = best_reviews
    report = best_report
    
    print(f"\n✅ Final Quality Score: {best_score:.1f}/100")
    
    # Save final reports
    quality_reporter.save_report(report, report_path_json)
    quality_reporter.save_report_markdown(report, report_path_md)
    
    print(f"📄 Final report saved to: {report_path_md}")
    print(f"📄 Final JSON report saved to: {report_path_json}")
    
    # Generate comparison plots
    if sampled_real_reviews:
        print(f"📊 Generating comparison plots...")
        quality_reporter.generate_comparison_plots(reviews, sampled_real_reviews, report_dir)
        print(f"✅ Plots saved to: {report_dir}/")
    
    # Generate distribution analysis plots
    print(f"📈 Generating distribution analysis...")
    quality_reporter.generate_distribution_plots(reviews, report_dir)
    print(f"✅ Distribution analysis saved to: {report_dir}/distribution_analysis.png")
    
    # Print summary
    quality_reporter.print_summary(report)
    
    return report
