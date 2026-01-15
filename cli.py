"""
Command-line interface for synthetic review generator.
"""
import click
import os
from dotenv import load_dotenv
from generate import generate_reviews, load_config
from compare import compare_reviews, print_statistics


@click.group()
def cli():
    """Synthetic Review Generator - Generate and analyze synthetic product reviews."""
    load_dotenv()


@cli.command()
@click.option('--num-reviews', '-n', type=int, help='Number of reviews to generate')
@click.option('--provider', '-p', type=click.Choice(['huggingface', 'gemini']), help='LLM provider to use')
@click.option('--config', '-c', default='config/generation_config.yaml', help='Path to config file')
@click.option('--output', '-o', default='data/generated_reviews.jsonl', help='Output file path')
def generate(num_reviews, provider, config, output):
    """Generate synthetic reviews using LLM."""
    
    # Load defaults from config if not specified
    cfg = load_config(config)
    
    if num_reviews is None:
        num_reviews = cfg.get('num_reviews', 100)
    
    if provider is None:
        provider = os.getenv('DEFAULT_PROVIDER', 'huggingface')

    
    click.echo(f"\n🚀 Starting generation...")
    click.echo(f"   Reviews: {num_reviews}")
    click.echo(f"   Provider: {provider}")
    click.echo(f"   Config: {config}")
    click.echo(f"   Output: {output}\n")
    
    try:
        reviews = generate_reviews(
            num_reviews=num_reviews,
            provider=provider,
            config_path=config,
            output_path=output
        )
        click.echo(f"\n✅ Generation complete! Generated {len(reviews)} reviews.")
    except Exception as e:
        click.echo(f"\n❌ Error: {str(e)}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--synthetic', '-s', default='data/generated_reviews.jsonl', help='Path to synthetic reviews')
@click.option('--real', '-r', default='data/real_reviews.csv', help='Path to real reviews')
@click.option('--config', '-c', default='config/generation_config.yaml', help='Path to config file')
@click.option('--output', '-o', default='quality_report.json', help='Output report path')
def analyze(synthetic, real, config, output):
    """Analyze quality of synthetic reviews."""
    
    click.echo(f"\n📊 Starting quality analysis...")
    click.echo(f"   Synthetic: {synthetic}")
    click.echo(f"   Real: {real}")
    click.echo(f"   Output: {output}\n")
    
    try:
        report = compare_reviews(
            synthetic_path=synthetic,
            real_path=real,
            config_path=config,
            output_path=output
        )
        
        score = report['quality_score']['overall']
        grade = report['quality_score']['grade']
        click.echo(f"\n✅ Analysis complete! Quality Score: {score}/100 ({grade})")
        
    except Exception as e:
        click.echo(f"\n❌ Error: {str(e)}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--file', '-f', default='data/generated_reviews.jsonl', help='Path to reviews file')
def stats(file):
    """Show statistics for generated reviews."""
    
    try:
        print_statistics(file)
    except Exception as e:
        click.echo(f"\n❌ Error: {str(e)}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--num-reviews', '-n', type=int, help='Number of reviews to generate')
@click.option('--provider', '-p', type=click.Choice(['huggingface', 'gemini']), help='LLM provider to use')
@click.option('--config', '-c', default='config/generation_config.yaml', help='Path to config file')
def run(num_reviews, provider, config):
    """Generate reviews and analyze quality (full pipeline)."""
    
    click.echo("\n🔄 Running full pipeline: Generate → Analyze\n")
    
    # Generate
    cfg = load_config(config)
    if num_reviews is None:
        num_reviews = cfg.get('num_reviews', 100)
    if provider is None:
        provider = os.getenv('DEFAULT_PROVIDER', 'huggingface')

    
    try:
        # Step 1: Generate
        click.echo("Step 1/2: Generating reviews...")
        reviews = generate_reviews(
            num_reviews=num_reviews,
            provider=provider,
            config_path=config
        )
        
        # Step 2: Analyze
        click.echo("\nStep 2/2: Analyzing quality...")
        report = compare_reviews(config_path=config)
        
        score = report['quality_score']['overall']
        grade = report['quality_score']['grade']
        
        click.echo(f"\n✅ Pipeline complete!")
        click.echo(f"   Generated: {len(reviews)} reviews")
        click.echo(f"   Quality Score: {score}/100 ({grade})")
        
    except Exception as e:
        click.echo(f"\n❌ Error: {str(e)}", err=True)
        raise click.Abort()


if __name__ == '__main__':
    cli()
