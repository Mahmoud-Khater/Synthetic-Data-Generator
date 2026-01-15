# LLM Provider Migration Guide

## Changes Made

Successfully replaced the LLM providers:

### Before
- ❌ OpenAI (GPT-4o-mini) - Paid API
- ❌ Anthropic (Claude 3.5 Sonnet) - Paid API

### After
- ✅ **Hugging Face** (Mistral-7b-FashionAssistant) - Free/Open Source
- ✅ **Google Gemini** (gemini-1.5-flash) - Free tier available

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `transformers` - Hugging Face models
- `torch` - PyTorch for model inference
- `accelerate` - Faster model loading
- `google-generativeai` - Gemini API

### 2. Configure API Keys

Create a `.env` file:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# Hugging Face Configuration (Optional - for private models)
HUGGINGFACE_TOKEN=your_huggingface_token_here

# Google Gemini Configuration (Required)
GEMINI_API_KEY=your_gemini_api_key_here

# Default provider
DEFAULT_PROVIDER=gemini  # or 'huggingface'
```

### 3. Get API Keys

#### Hugging Face Token (Optional)
- Visit: https://huggingface.co/settings/tokens
- Create a new token
- The Mistral-7b-FashionAssistant model is public, so token may not be required

#### Gemini API Key (Free)
- Visit: https://makersuite.google.com/app/apikey
- Click "Create API Key"
- Free tier includes generous limits

## Usage

### Using Gemini (Recommended for Quick Start)

```bash
# Generate reviews with Gemini
python cli.py generate -p gemini -n 10

# Run full pipeline
python cli.py run -p gemini
```

### Using Hugging Face

```bash
# Generate reviews with Hugging Face
python cli.py generate -p huggingface -n 10

# Note: First run will download the model (~14GB)
```

## Model Comparison

| Feature | Hugging Face (Mistral-7b) | Google Gemini (1.5 Flash) |
|---------|---------------------------|---------------------------|
| **Cost** | Free (runs locally) | Free tier available |
| **Speed** | Slower (local inference) | Fast (API) |
| **Setup** | Downloads ~14GB model | Just API key |
| **Privacy** | Fully local | Cloud-based |
| **GPU** | Recommended | Not needed |
| **Quality** | Fashion-specialized | General purpose |

## Recommendations

### For Quick Testing
Use **Gemini**:
- Fast setup (just API key)
- No model download
- Good quality
- Free tier sufficient for testing

### For Production/Privacy
Use **Hugging Face**:
- No API costs
- Data stays local
- Fashion-specialized model
- Requires GPU for good performance

## Example

```python
from generate import generate_reviews

# Using Gemini (fast)
reviews = generate_reviews(
    num_reviews=10,
    provider='gemini'
)

# Using Hugging Face (local)
reviews = generate_reviews(
    num_reviews=10,
    provider='huggingface'
)
```

## Troubleshooting

### Hugging Face Issues

**Model download fails:**
```bash
# Check disk space (need ~14GB)
df -h

# Try manual download
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('Soorya03/Mistral-7b-FashionAssistant')"
```

**Out of memory:**
```python
# The model will automatically use CPU if no GPU available
# For better performance, use a machine with GPU (8GB+ VRAM)
```

### Gemini Issues

**API key error:**
```bash
# Verify your API key is set
echo $GEMINI_API_KEY

# Or check .env file
cat .env
```

**Rate limit:**
```
# Free tier limits:
# - 15 requests per minute
# - 1 million tokens per minute
# Add delays between requests if needed
```

## Performance Tips

### Hugging Face
- Use GPU for 10x faster generation
- First run downloads model (one-time)
- Subsequent runs are faster
- Consider batch processing

### Gemini
- Very fast API responses
- No local resources needed
- Respects rate limits automatically
- Good for quick iterations

## Next Steps

1. Test with small batch (10 reviews)
2. Compare quality between providers
3. Choose based on your needs:
   - Speed → Gemini
   - Privacy → Hugging Face
   - Cost → Both are free!
