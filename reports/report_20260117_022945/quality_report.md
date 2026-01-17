# Synthetic Review Quality Report

## Metadata
- **Generated**: 2026-01-17T02:29:51.255926
- **Provider**: unknown
- **Synthetic Reviews**: 5
- **Real Reviews**: 3

---

## Quality Score

**Overall**: 42.0/100 - **F (Unacceptable)**

| Metric | Score |
|--------|-------|
| Diversity | 52.64/100 |
| Bias | 25.00/100 |
| Realism | 50.00/100 |

---

## Diversity Analysis

### Vocabulary
- **Type-Token Ratio**: 0.670
- **Unique Tokens**: 181
- **Total Tokens**: 270
- **Avg Unique per Review**: 49.40

### Semantic Similarity
- **Average**: 0.618
- **Max**: 0.811
- **Min**: 0.399
- **Std Dev**: 0.1407

---

## Bias Detection

### Rating Distribution
| Rating | Actual | Expected | Deviation |
|--------|--------|----------|-----------|
| 0 | 0.0% | 10.0% | 10.0% |
| 1 | 20.0% | 15.0% | 5.0% |
| 2 | 40.0% | 30.0% | 10.0% |
| 3 | 20.0% | 25.0% | 5.0% |
| 4 | 20.0% | 20.0% | 0.0% |

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
- "a reliable choice" - 2 times (40.0%)
- "reliable choice for" - 2 times (40.0%)
- "a reliable choice for" - 2 times (40.0%)
- "walking these shoes" - 1 times (20.0%)
- "these shoes has" - 1 times (20.0%)
- "shoes has been" - 1 times (20.0%)
- "been a real" - 1 times (20.0%)
- "a real letdown" - 1 times (20.0%)
- "real letdown for" - 1 times (20.0%)
- "letdown for my" - 1 times (20.0%)

---

## Realism Analysis

### Aspect Coverage
- **Coverage Rate**: 60.0%
- **Avg Mentions per Aspect**: 0.86
- **Has Good Coverage**: No

**Aspect Mentions**:
- **comfort**: 2
- **fit**: 1
- **quality**: 1
- **style**: 0
- **durability**: 0
- **price**: 0
- **sizing**: 2

### Readability
- **Avg Flesch Score**: 52.51
- **Interpretation**: Fairly Difficult
- **Is Natural**: Yes

### AI Patterns
- **AI Pattern Rate**: 0.0%
- **Is Realistic**: Yes

---

## Comparison with Real Reviews

| Metric | Synthetic | Real | Difference |
|--------|-----------|------|------------|
| Vocabulary (TTR) | 0.670 | 0.700 | 0.030 |
| Avg Length (words) | 54.2 | 40.7 | 13.5 |
| Similarity to Real | 0.507 | - | - |

---

## Domain Validation

- **Shoe-Related**: 5/5 (100.0%)
- **Avg Relevance Score**: 0.988
- **Avg Keyword Count**: 8.6
- **Flagged Reviews**: 0

---

## Recommendations

1. Rating distribution deviates from expected. Adjust generation parameters or sampling.
2. Repetitive phrases detected. Increase generation diversity or use more varied prompts.
3. Low product aspect coverage. Emphasize relevant aspects in generation prompts.
4. Low personal pronoun usage. Encourage first-person perspective in prompts.
