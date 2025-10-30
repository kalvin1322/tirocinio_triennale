"""
Main CLI entry point for CT Reconstruction Training & Benchmarking Library
"""
import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from pathlib import Path

from .interactive import run_interactive_mode
from .commands import train_cmd, test_cmd, benchmark_cmd

app = typer.Typer(
    name="ct-benchmark",
    help="CT Reconstruction Training & Benchmarking Tool",
    add_completion=False
)
console = Console()


@app.command()
def interactive():
    """
    🎯 Launch interactive mode with guided setup (Recommended for beginners)
    """
    console.print(Panel.fit(
        "[bold cyan]Welcome to CT Reconstruction Training Tool![/bold cyan]\n"
        "This interactive mode will guide you through the process.",
        title="Interactive Mode"
    ))
    run_interactive_mode()


@app.command()
def train(
    model: str = typer.Option(..., "--model", "-m", help="Model name (UNet_V1, LPP_model)"),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Path to dataset"),
    epochs: int = typer.Option(5, "--epochs", "-e", help="Number of epochs"),
    batch_size: int = typer.Option(8, "--batch-size", "-b", help="Batch size"),
    lr: float = typer.Option(1e-4, "--learning-rate", "-lr", help="Learning rate"),
    output: str = typer.Option("outputs/trained_models", "--output", "-o", help="Output directory")
):
    """
    🚀 Train a model with specified parameters
    """
    train_cmd(model, dataset, epochs, batch_size, lr, output)


@app.command()
def test(
    model: str = typer.Option(..., "--model", "-m", help="Model name"),
    checkpoint: str = typer.Option(..., "--checkpoint", "-c", help="Path to model checkpoint"),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Path to test dataset"),
    output: str = typer.Option("outputs/test_results", "--output", "-o", help="Output directory")
):
    """
    🧪 Test a trained model on test dataset
    """
    test_cmd(model, checkpoint, dataset, output)


@app.command()
def benchmark(
    models: list[str] = typer.Option(..., "--models", "-m", help="Model names to compare"),
    checkpoints: list[str] = typer.Option(..., "--checkpoints", "-c", help="Paths to checkpoints"),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Path to test dataset"),
    output: str = typer.Option("outputs/benchmarks", "--output", "-o", help="Output directory")
):
    """
    📊 Benchmark multiple models and compare results
    """
    benchmark_cmd(models, checkpoints, dataset, output)


@app.command()
def wizard():
    """
    🧙 Launch setup wizard for first-time configuration
    """
    from .wizard import run_wizard
    run_wizard()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """
    CT Reconstruction Training & Benchmarking Tool
    
    Run without arguments to see available commands.
    Use 'interactive' mode for guided experience.
    """
    if ctx.invoked_subcommand is None:
        console.print(Panel.fit(
            "[bold green]CT Reconstruction Training Tool[/bold green]\n\n"
            "Available modes:\n"
            "  • [cyan]interactive[/cyan] - Guided mode (recommended)\n"
            "  • [cyan]wizard[/cyan] - First-time setup\n"
            "  • [cyan]train[/cyan] - Train a model\n"
            "  • [cyan]test[/cyan] - Test a model\n"
            "  • [cyan]benchmark[/cyan] - Compare models\n\n"
            "Run [yellow]ct-benchmark --help[/yellow] for more info",
            title="Welcome"
        ))
        console.print("\n[dim]Tip: Run 'ct-benchmark interactive' to get started[/dim]\n")


if __name__ == "__main__":
    app()
