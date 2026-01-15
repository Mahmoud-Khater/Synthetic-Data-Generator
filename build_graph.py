from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, Dict, List
from models.azure_openai_generator import AzureOpenAIGenerator
from models.azure_openai_reviewer import AzureOpenAIReviewer
from quality.quality_report import QualityReporter
import json
from datetime import datetime
from dotenv import load_dotenv

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


# ----- 2. Load Azure OpenAI Generator & Reviewer -----
config = {
    "generation_params": {
        "max_tokens": 200,
        "temperature": 0.8,
        "top_p": 0.9
    },
    "review_params": {
        "max_tokens": 1000,
        "temperature": 0.3,
        "top_p": 0.9
    },
    "product_context": {
        "category": "product",
        "aspects": ["quality", "price", "service", "experience"]
    },
    "rating_distribution": {
        1: 0.05,
        2: 0.10,
        3: 0.20,
        4: 0.35,
        5: 0.30
    }
}

generator = AzureOpenAIGenerator(config=config)
reviewer = AzureOpenAIReviewer(config=config)
quality_reporter = QualityReporter(config=config)


# ----- 3. Node: Generate Review -----
def generate_review_node(state: ReviewState) -> ReviewState:
    """Generate a review using Azure OpenAI."""
    print(f"\n🔄 Attempt {state['attempt'] + 1}/{state['max_attempts']}: Generating review...")
    
    review = generator.generate_review(
        rating=state["rating"],
        persona=state["persona"],
        real_reviews=[]  # Can pass real reviews for context if available
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
        
        return {**state, "quality_assessment": assessment}
        
    except Exception as e:
        print(f"⚠️  Error during review: {str(e)}")
        # If review fails, mark as failed assessment
        return {
            **state,
            "quality_assessment": {
                "pass": False,
                "overall_score": 0,
                "issues": [f"Review error: {str(e)}"]
            }
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
        print(f"⚠️  Max attempts ({state['max_attempts']}) reached. Accepting review.")
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
        rating: Rating (1-5)
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
        "all_reviews": []
    }
    
    print(f"\n{'='*60}")
    print(f"Starting review generation for {persona.get('name', 'Unknown')} - Rating: {rating}/5")
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
    import random
    
    reviews = []
    
    for i in range(num_reviews):
        # Select random persona and rating
        persona = random.choice(personas)
        rating = generator.select_rating()
        
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
    
    return reviews


def generate_quality_report(reviews: List[Dict], output_path: str = "quality_report.json"):
    """
    Generate a comprehensive quality report for the generated reviews.
    
    Args:
        reviews: List of review dictionaries
        output_path: Path to save the report
        
    Returns:
        Quality report dictionary
    """
    print(f"\n{'='*60}")
    print("Generating Quality Report")
    print(f"{'='*60}\n")
    
    report = quality_reporter.generate_report(reviews)
    
    # Save report
    quality_reporter.save_report(report, output_path)
    print(f"📄 Report saved to: {output_path}")
    
    # Print summary
    quality_reporter.print_summary(report)
    
    return report
