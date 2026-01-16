"""
Test script to demonstrate the best score selection feature.
This simulates a scenario where all 3 attempts fail but have different scores.
"""
from build_graph import run_review_generation

# Example persona
test_persona = {
    "name": "Test User",
    "description": "A test user for demonstrating the best score selection",
    "traits": [
        "Writes detailed reviews",
        "Focuses on quality",
        "Critical but fair"
    ]
}

print("\n" + "="*80)
print("TESTING: Best Score Selection When All Attempts Fail")
print("="*80)
print("\nThis test will run the workflow and show how it selects the best review")
print("from all attempts if none pass the quality threshold.\n")

# Run the workflow
result = run_review_generation(
    persona=test_persona,
    rating=4,
    max_attempts=3
)

print("\n" + "="*80)
print("FINAL RESULT")
print("="*80)

# Show attempt history
if "attempt_history" in result and result["attempt_history"]:
    print(f"\n📊 Attempt History:")
    for attempt in result["attempt_history"]:
        print(f"   Attempt {attempt['attempt_number']}: Score {attempt['overall_score']}/10")
    
    # Find the best score
    best = max(result["attempt_history"], key=lambda x: x["overall_score"])
    print(f"\n🏆 Best Score: {best['overall_score']}/10 (Attempt {best['attempt_number']})")

print(f"\n📝 Final Review (selected):")
print(f"   {result['review'][:200]}...")
print(f"\n✅ Final Quality Score: {result['quality_assessment'].get('overall_score', 0)}/10")
print(f"🔄 Total Attempts: {result['attempt']}/3")
print(f"✔️  Passed: {result['quality_assessment'].get('pass', False)}")

print("\n" + "="*80 + "\n")
