# Synthetic Review Quality Report

## Metadata
- **Generated**: 2026-01-16T23:09:28.875097
- **Provider**: azure_openai_gpt-4.1-mini
- **Synthetic Reviews**: 10
- **Real Reviews**: 8

---

## Quality Score

**Overall**: 53.9/100 - **F (Unacceptable)**

| Metric | Score |
|--------|-------|
| Diversity | 39.68/100 |
| Bias | 45.00/100 |
| Realism | 75.00/100 |

---

## Diversity Analysis

### Vocabulary
- **Type-Token Ratio**: 0.482
- **Unique Tokens**: 507
- **Total Tokens**: 1052
- **Avg Unique per Review**: 88.80

### Semantic Similarity
- **Average**: 0.688
- **Max**: 0.850
- **Min**: 0.495
- **Std Dev**: 0.0781

---

## Bias Detection

### Rating Distribution
| Rating | Actual | Expected | Deviation |
|--------|--------|----------|-----------|
| 0 | 0.0% | 0.0% | 0.0% |
| 1 | 0.0% | 0.0% | 0.0% |
| 2 | 0.0% | 0.0% | 0.0% |
| 3 | 0.0% | 0.0% | 0.0% |
| 4 | 0.0% | 0.0% | 0.0% |

**Total Deviation**: 10.0%  
**Biased**: No

### Sentiment Consistency
- **Inconsistencies**: 0/10
- **Inconsistency Rate**: 0.0%
- **Consistent**: Yes

### Repetitive Patterns
- **Unique Reviews**: 10/10
- **Duplicate Rate**: 0.0%
- **Has Repetition Issues**: Yes

**Top Repetitive Phrases**:
- "into these shoes" - 3 times (30.0%)
- "the materials feel" - 3 times (30.0%)
- "cheap and flimsy," - 2 times (20.0%)
- "the cushioning barely" - 2 times (20.0%)
- "not built to" - 2 times (20.0%)
- "stepping into these" - 2 times (20.0%)
- "the material feels" - 2 times (20.0%)
- "the sizing runs" - 2 times (20.0%)
- "sizing runs oddly" - 2 times (20.0%)
- "short in both" - 2 times (20.0%)

---

## Realism Analysis

### Aspect Coverage
- **Coverage Rate**: 100.0%
- **Avg Mentions per Aspect**: 3.71
- **Has Good Coverage**: Yes

**Aspect Mentions**:
- **comfort**: 8
- **fit**: 1
- **quality**: 3
- **style**: 0
- **durability**: 7
- **price**: 0
- **sizing**: 7

### Readability
- **Avg Flesch Score**: 45.65
- **Interpretation**: Difficult
- **Is Natural**: No

### AI Patterns
- **AI Pattern Rate**: 0.0%
- **Is Realistic**: Yes

---

## Comparison with Real Reviews

| Metric | Synthetic | Real | Difference |
|--------|-----------|------|------------|
| Vocabulary (TTR) | 0.482 | 0.504 | 0.022 |
| Avg Length (words) | 105.2 | 57.1 | 48.1 |
| Similarity to Real | 0.640 | - | - |

---

## Domain Validation

- **Shoe-Related**: 10/10 (100.0%)
- **Avg Relevance Score**: 0.902
- **Avg Keyword Count**: 11.1
- **Flagged Reviews**: 0

---

## Recommendations

1. Repetitive phrases detected. Increase generation diversity or use more varied prompts.
2. Readability scores outside natural range. Adjust language complexity in prompts.
