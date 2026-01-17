# Synthetic Review Quality Report

## Metadata
- **Generated**: 2026-01-16T23:15:35.820158
- **Provider**: azure_openai_gpt-4.1-mini
- **Synthetic Reviews**: 5
- **Real Reviews**: 3

---

## Quality Score

**Overall**: 56.3/100 - **F (Unacceptable)**

| Metric | Score |
|--------|-------|
| Diversity | 41.80/100 |
| Bias | 25.00/100 |
| Realism | 100.00/100 |

---

## Diversity Analysis

### Vocabulary
- **Type-Token Ratio**: 0.554
- **Unique Tokens**: 311
- **Total Tokens**: 561
- **Avg Unique per Review**: 92.80

### Semantic Similarity
- **Average**: 0.718
- **Max**: 0.791
- **Min**: 0.653
- **Std Dev**: 0.0453

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

**Total Deviation**: 30.0%  
**Biased**: Yes

### Sentiment Consistency
- **Inconsistencies**: 0/5
- **Inconsistency Rate**: 0.0%
- **Consistent**: Yes

### Repetitive Patterns
- **Unique Reviews**: 5/5
- **Duplicate Rate**: 0.0%
- **Has Repetition Issues**: Yes

**Top Repetitive Phrases**:
- "hoping for a" - 1 times (20.0%)
- "for a solid" - 1 times (20.0%)
- "a solid bargain," - 1 times (20.0%)
- "solid bargain, i" - 1 times (20.0%)
- "bargain, i gave" - 1 times (20.0%)
- "i gave these" - 1 times (20.0%)
- "gave these shoes" - 1 times (20.0%)
- "these shoes a" - 1 times (20.0%)
- "shoes a try," - 1 times (20.0%)
- "try, but they" - 1 times (20.0%)

---

## Realism Analysis

### Aspect Coverage
- **Coverage Rate**: 100.0%
- **Avg Mentions per Aspect**: 2.43
- **Has Good Coverage**: Yes

**Aspect Mentions**:
- **comfort**: 5
- **fit**: 0
- **quality**: 2
- **style**: 1
- **durability**: 4
- **price**: 1
- **sizing**: 4

### Readability
- **Avg Flesch Score**: 51.72
- **Interpretation**: Fairly Difficult
- **Is Natural**: Yes

### AI Patterns
- **AI Pattern Rate**: 0.0%
- **Is Realistic**: Yes

---

## Comparison with Real Reviews

| Metric | Synthetic | Real | Difference |
|--------|-----------|------|------------|
| Vocabulary (TTR) | 0.554 | 0.786 | 0.231 |
| Avg Length (words) | 110.8 | 18.7 | 92.1 |
| Similarity to Real | 0.426 | - | - |

---

## Domain Validation

- **Shoe-Related**: 5/5 (100.0%)
- **Avg Relevance Score**: 1.000
- **Avg Keyword Count**: 14.2
- **Flagged Reviews**: 0

---

## Recommendations

1. Reviews are too similar. Consider increasing temperature or using more diverse personas.
2. Rating distribution deviates from expected. Adjust generation parameters or sampling.
3. Repetitive phrases detected. Increase generation diversity or use more varied prompts.
