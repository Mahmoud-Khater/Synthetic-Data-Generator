# Azure OpenAI Review Generation Workflow

This document describes the Azure OpenAI-based review generation workflow with quality checking and automatic regeneration.

## Overview

The workflow uses **LangGraph** to orchestrate a generate-review-regenerate loop:

1. **Generate** a review using Azure OpenAI
2. **Review** the quality using Azure OpenAI Reviewer
3. **Regenerate** if quality is poor (max 3 attempts)
4. **Report** comprehensive quality metrics

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Review Generation Flow                    │
└─────────────────────────────────────────────────────────────┘

    ┌──────────────┐
    │    START     │
    └──────┬───────┘
           │
           ▼
    ┌──────────────────┐
    │ Generate Review  │ ◄─────┐
    │ (Azure OpenAI)   │       │
    └──────┬───────────┘       │
           │                   │
           ▼                   │
    ┌──────────────────┐       │
    │  Quality Check   │       │
    │ (Azure Reviewer) │       │
    └──────┬───────────┘       │
           │                   │
           ▼                   │
    ┌──────────────────┐       │
    │  Pass Quality?   │───No──┤ (Retry if attempts < 3)
    └──────┬───────────┘       │
           │ Yes               │
           ▼                   │
    ┌──────────────┐           │
    │     END      │           │
    └──────────────┘           │
```

## Components

### 1. Azure OpenAI Generator (`models/azure_openai_generator.py`)
- Generates synthetic reviews based on personas and ratings
- Uses configurable temperature and token limits
- Inherits from `BaseGenerator`

### 2. Azure OpenAI Reviewer (`models/azure_openai_reviewer.py`)
- Evaluates generated reviews for quality
- Provides structured assessment with scores (1-10):
  - Authenticity
  - Consistency (rating alignment)
  - Persona alignment
  - Quality (writing)
  - Realism
  - Overall score
- Returns pass/fail decision with issues and suggestions

### 3. LangGraph Workflow (`build_graph.py`)
- Orchestrates the generate-review-regenerate loop
- Maximum 3 regeneration attempts
- Tracks quality assessments and attempt counts

### 4. Quality Reporter (`quality/quality_report.py`)
- Generates comprehensive quality reports
- Analyzes diversity, bias, and realism
- Provides actionable recommendations

## Usage

### Single Review Generation

```python
from build_graph import run_review_generation

persona = {
    "name": "Tech Enthusiast",
    "demographics": "25-35, tech-savvy",
    "preferences": "Values innovation",
    "writing_style": "Detailed, technical"
}

result = run_review_generation(
    persona=persona,
    rating=5,
    max_attempts=3
)

print(result['review'])
print(f"Quality Score: {result['quality_assessment']['overall_score']}/10")
print(f"Attempts: {result['attempt']}")
```

### Batch Generation with Quality Report

```python
from build_graph import generate_batch_reviews, generate_quality_report

# Generate 10 reviews
reviews = generate_batch_reviews(
    personas=PERSONAS,
    num_reviews=10,
    max_attempts=3
)

# Generate quality report
report = generate_quality_report(
    reviews=reviews,
    output_path="data/quality_report.json"
)
```

### Run Example Script

```bash
# Run batch generation example (default)
python example_workflow.py

# Run single review example
python example_workflow.py 1

# Run custom workflow example
python example_workflow.py 3
```

## Configuration

The workflow uses a configuration dictionary:

```python
config = {
    "generation_params": {
        "max_tokens": 200,
        "temperature": 0.8,  # Higher for creative generation
        "top_p": 0.9
    },
    "review_params": {
        "max_tokens": 1000,
        "temperature": 0.3,  # Lower for consistent evaluation
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
```

## Environment Variables

Required environment variables (set in `.env`):

```bash
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-08-01-preview
```

## Quality Assessment Criteria

The reviewer evaluates each generated review on:

1. **Authenticity (1-10)**: Does it sound natural and human-written?
2. **Consistency (1-10)**: Does content match the rating?
3. **Persona Alignment (1-10)**: Does it reflect the persona's characteristics?
4. **Quality (1-10)**: Is it well-written and coherent?
5. **Realism (1-10)**: Could this appear on a real review platform?
6. **Overall Score (1-10)**: Aggregate assessment
7. **Pass/Fail**: Boolean decision (typically pass if overall > 6)

## Quality Report

The quality report includes:

- **Diversity Analysis**: Vocabulary richness, semantic similarity
- **Bias Analysis**: Rating distribution, sentiment consistency, repetitive patterns
- **Realism Analysis**: Aspect coverage, readability, AI patterns, pronoun usage
- **Overall Quality Score**: Weighted average with letter grade
- **Recommendations**: Actionable suggestions for improvement

## Output Format

Generated reviews are saved with metadata:

```json
{
  "text": "The review text...",
  "rating": 5,
  "persona": {...},
  "provider": "azure_openai_gpt-4o-mini",
  "attempts": 1,
  "quality_assessment": {
    "overall_score": 8,
    "pass": true,
    "authenticity_score": 8,
    "consistency_score": 9,
    "persona_alignment_score": 7,
    "quality_score": 8,
    "realism_score": 8,
    "issues": [],
    "suggestions": []
  },
  "generated_at": "2026-01-15T22:37:00"
}
```

## Benefits

1. **Quality Assurance**: Automatic quality checking ensures high-quality output
2. **Efficiency**: Regenerates only when needed (max 3 attempts)
3. **Transparency**: Detailed quality assessments and reports
4. **Flexibility**: Configurable parameters and personas
5. **Comprehensive**: End-to-end workflow from generation to reporting

## Next Steps

- Adjust `max_attempts` based on your quality requirements
- Customize personas for your specific use case
- Tune temperature and other generation parameters
- Add real reviews for style reference
- Integrate with your data pipeline
