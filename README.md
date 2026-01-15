# Synthetic Review Generator

A comprehensive system for generating high-quality synthetic product reviews using Large Language Models (LLMs) with built-in quality metrics and bias detection.

## 🌟 Features

- **Multi-Provider Support**: Generate reviews using OpenAI GPT or Anthropic Claude
- **Persona-Based Generation**: Create diverse reviews from different customer perspectives
- **Quality Metrics**: Comprehensive analysis including:
  - Diversity metrics (vocabulary overlap, semantic similarity)
  - Bias detection (sentiment skew, repetitive patterns)
  - Realism checks (domain relevance, AI pattern detection)
- **Comparison Tools**: Compare synthetic reviews against real reviews
- **CLI Interface**: Easy-to-use command-line tools
- **Configurable**: Flexible YAML configuration for personas, rating distribution, and generation parameters

## 📁 Project Structure

```
synthetic-review-generator/
│
├── data/
│   ├── real_reviews.csv         # 30 real shoe reviews (sample data)
│   └── generated_reviews.jsonl  # Generated synthetic reviews
│
├── config/
│   └── generation_config.yaml   # Configuration for personas, ratings, etc.
│
├── models/
│   ├── base_generator.py        # Base generator interface
│   ├── openai_generator.py      # OpenAI GPT generator
│   └── anthropic_generator.py   # Anthropic Claude generator
│
├── quality/
│   ├── diversity.py             # Vocabulary & semantic diversity metrics
│   ├── bias.py                  # Bias and pattern detection
│   ├── realism.py               # Domain realism checks
│   └── quality_report.py        # Comprehensive quality reporting
│
├── cli.py                       # Command-line interface
├── generate.py                  # Core generation logic
├── compare.py                   # Real vs synthetic comparison
│
├── README.md                    # This file
├── requirements.txt             # Python dependencies
└── .env.example                 # Environment variable template
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to the project directory
cd EasyGenerator

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic Configuration
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Default settings
DEFAULT_PROVIDER=openai
OPENAI_MODEL=gpt-4o-mini
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
```

### 3. Generate Reviews

```bash
# Generate 100 reviews using default settings
python cli.py generate

# Generate 50 reviews using Anthropic
python cli.py generate -n 50 -p anthropic

# Run full pipeline (generate + analyze)
python cli.py run
```

## 📖 Usage Guide

### CLI Commands

#### Generate Reviews

```bash
python cli.py generate [OPTIONS]

Options:
  -n, --num-reviews INTEGER       Number of reviews to generate
  -p, --provider [openai|anthropic]  LLM provider to use
  -c, --config PATH               Path to config file
  -o, --output PATH               Output file path
```

#### Analyze Quality

```bash
python cli.py analyze [OPTIONS]

Options:
  -s, --synthetic PATH  Path to synthetic reviews
  -r, --real PATH       Path to real reviews
  -c, --config PATH     Path to config file
  -o, --output PATH     Output report path
```

#### Show Statistics

```bash
python cli.py stats [OPTIONS]

Options:
  -f, --file PATH  Path to reviews file
```

#### Run Full Pipeline

```bash
python cli.py run [OPTIONS]

Options:
  -n, --num-reviews INTEGER       Number of reviews to generate
  -p, --provider [openai|anthropic]  LLM provider to use
  -c, --config PATH               Path to config file
```

### Python API

```python
from generate import generate_reviews
from compare import compare_reviews

# Generate reviews
reviews = generate_reviews(
    num_reviews=100,
    provider='openai',
    config_path='config/generation_config.yaml',
    output_path='data/generated_reviews.jsonl'
)

# Analyze quality
report = compare_reviews(
    synthetic_path='data/generated_reviews.jsonl',
    real_path='data/real_reviews.csv',
    output_path='quality_report.json'
)

print(f"Quality Score: {report['quality_score']['overall']}/100")
```

## ⚙️ Configuration

Edit `config/generation_config.yaml` to customize:

### Number of Reviews

```yaml
num_reviews: 100
```

### Rating Distribution

```yaml
rating_distribution:
  1: 0.05  # 5% - Very negative
  2: 0.10  # 10% - Negative
  3: 0.20  # 20% - Neutral
  4: 0.35  # 35% - Positive
  5: 0.30  # 30% - Very positive
```

### Personas

```yaml
personas:
  - name: "casual_buyer"
    description: "Someone who buys shoes occasionally for everyday use"
    traits:
      - "focuses on comfort and price"
      - "uses simple language"
      - "mentions daily activities"
```

### Generation Parameters

```yaml
generation_params:
  temperature: 0.8    # Higher = more creative
  max_tokens: 200     # Maximum review length
  top_p: 0.9         # Nucleus sampling parameter
```

## 📊 Quality Metrics

### Diversity Metrics

- **Type-Token Ratio (TTR)**: Vocabulary diversity
- **Semantic Similarity**: How similar reviews are to each other
- **TF-IDF Similarity**: Content overlap analysis

### Bias Detection

- **Rating Distribution**: Deviation from expected distribution
- **Sentiment Consistency**: Rating-sentiment alignment
- **Repetitive Patterns**: Detection of overused phrases
- **Length Distribution**: Uniformity checks

### Realism Checks

- **Aspect Coverage**: Mentions of relevant product features
- **Readability**: Natural language complexity
- **AI Pattern Detection**: Identification of AI-generated markers
- **Personal Pronoun Usage**: First-person perspective usage

### Quality Score

The overall quality score (0-100) is calculated as:
- **30%** Diversity
- **35%** Bias (lower bias = higher score)
- **35%** Realism

Grades:
- **A (90-100)**: Excellent
- **B (80-89)**: Good
- **C (70-79)**: Acceptable
- **D (60-69)**: Poor
- **F (<60)**: Unacceptable

## 📝 Example Output

### Generated Review

```json
{
  "id": 1,
  "rating": 5,
  "text": "These shoes are absolutely fantastic! I wear them to the gym every day and they provide amazing support during my workouts. The cushioning is perfect for running and the grip is excellent. They're also surprisingly lightweight. Best athletic shoes I've owned!",
  "persona": "athlete",
  "provider": "openai_gpt-4o-mini"
}
```

### Quality Report Summary

```
==============================================================
SYNTHETIC REVIEW QUALITY REPORT
==============================================================

Generated: 2026-01-15T15:20:00
Provider: openai_gpt-4o-mini
Synthetic Reviews: 100
Real Reviews: 30

--------------------------------------------------------------
QUALITY SCORE
--------------------------------------------------------------
Overall Score: 85.5/100 - B (Good)
  - Diversity: 82.3/100
  - Bias: 87.5/100
  - Realism: 86.8/100

--------------------------------------------------------------
RECOMMENDATIONS
--------------------------------------------------------------
1. Quality metrics look good! No major issues detected.
==============================================================
```

## 🔧 Customization

### Adding New Personas

Edit `config/generation_config.yaml`:

```yaml
personas:
  - name: "your_persona_name"
    description: "Description of this persona"
    traits:
      - "trait 1"
      - "trait 2"
      - "trait 3"
```

### Changing Product Category

Update the product context in `config/generation_config.yaml`:

```yaml
product_context:
  category: "electronics"  # Change from "shoes"
  aspects:
    - "battery life"
    - "screen quality"
    - "performance"
    - "build quality"
```

### Adding New LLM Providers

1. Create a new generator in `models/`:

```python
from models.base_generator import BaseGenerator

class NewProviderGenerator(BaseGenerator):
    def generate_review(self, rating, persona, real_reviews):
        # Implementation
        pass
    
    def get_provider_name(self):
        return "new_provider"
```

2. Update `generate.py` to include the new provider

## 🐛 Troubleshooting

### API Key Errors

```
ValueError: OpenAI API key not provided
```

**Solution**: Ensure your `.env` file contains valid API keys

### Import Errors

```
ModuleNotFoundError: No module named 'sentence_transformers'
```

**Solution**: Install all dependencies: `pip install -r requirements.txt`

### Low Quality Scores

**Solutions**:
- Increase temperature for more diversity
- Add more varied personas
- Adjust prompts in `base_generator.py`
- Use a more capable model (e.g., GPT-4 instead of GPT-3.5)

## 📄 License

This project is open source and available for educational and commercial use.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional LLM providers
- More sophisticated quality metrics
- Multi-language support
- Fine-tuning capabilities
- Web interface

## 📧 Support

For issues or questions, please create an issue in the repository or contact the development team.
