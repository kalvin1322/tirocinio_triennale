"""
Interactive mode with guided menus for CT Reconstruction Training
"""
import inquirer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from pathlib import Path
import yaml
from datetime import datetime

console = Console()


def load_current_experiment():
    """Load the currently selected experiment configuration"""
    current_exp_file = Path(".current_experiment")
    if current_exp_file.exists():
        with open(current_exp_file, 'r') as f:
            exp_data = yaml.safe_load(f)
            return exp_data
    return None


def save_current_experiment(experiment_config):
    """Save the currently selected experiment"""
    current_exp_file = Path(".current_experiment")
    with open(current_exp_file, 'w') as f:
        yaml.dump(experiment_config, f)


def experiment_interactive():
    """Interactive experiment creation/selection"""
    console.clear()
    console.print(Panel.fit(
        "[bold magenta]Experiment Management[/bold magenta]",
        title="Experiments"
    ))
    
    # Check for existing experiments
    experiments_index = Path("experiments") / "experiments_index.yaml"
    existing_experiments = []
    
    if experiments_index.exists():
        with open(experiments_index, 'r') as f:
            index = yaml.safe_load(f)
            if index and 'experiments' in index:
                existing_experiments = index['experiments']
    
    choices = [
        ('➕ Create new experiment', 'new'),
    ]
    
    if existing_experiments:
        console.print(f"\n[cyan]Found {len(existing_experiments)} existing experiment(s):[/cyan]")
        for exp in existing_experiments[-5:]:  # Show last 5
            console.print(f"  • {exp['name']} - {exp.get('description', 'No description')}")
        console.print()
        choices.append(('📂 Select existing experiment', 'select'))
    
    choices.append(('← Back to Main Menu', 'back'))
    
    questions = [
        inquirer.List(
            'action',
            message="Choose an action",
            choices=choices
        )
    ]
    
    answers = inquirer.prompt(questions)
    
    if not answers or answers.get('action') == 'back':
        return None
    
    if answers['action'] == 'new':
        from .wizard import run_wizard
        config_path = run_wizard()
        if config_path:
            with open(config_path, 'r') as f:
                exp_config = yaml.safe_load(f)
            save_current_experiment(exp_config)
            return exp_config
    elif answers['action'] == 'select':
        # Show list of experiments to select
        exp_choices = [(f"{exp['name']} - {exp.get('description', '')}", exp) 
                       for exp in existing_experiments]
        exp_choices.append(('← Back', 'BACK'))
        
        select_questions = [
            inquirer.List(
                'experiment',
                message="Select an experiment",
                choices=exp_choices
            )
        ]
        
        select_answers = inquirer.prompt(select_questions)
        if select_answers and select_answers['experiment'] != 'BACK':
            exp = select_answers['experiment']
            config_path = Path(exp['config_path'])
            with open(config_path, 'r') as f:
                exp_config = yaml.safe_load(f)
            save_current_experiment(exp_config)
            exp_name = exp_config.get('experiment', {}).get('name', exp['name'])
            console.print(f"\n[green]✓ Selected experiment: {exp_name}[/green]")
            console.input("\n[dim]Press Enter to continue...[/dim]")
            return exp_config
    
    return load_current_experiment()


def run_interactive_mode():
    """Main interactive mode with menu navigation"""
    
    # Check for existing experiments
    current_experiment = load_current_experiment()
    
    while True:
        console.clear()
        console.print(Panel.fit(
            "[bold cyan]CT Reconstruction Training - Interactive Mode[/bold cyan]",
            title="Main Menu"
        ))
        
        # Show current experiment if set
        if current_experiment:
            exp_name = current_experiment.get('experiment', {}).get('name', 'Unknown')
            console.print(f"[dim]Current experiment: [cyan]{exp_name}[/cyan][/dim]\n")
        else:
            console.print(f"[yellow]⚠ No experiment selected. Create one first![/yellow]\n")
        
        questions = [
            inquirer.List(
                'action',
                message="What would you like to do?",
                choices=[
                    ('🔬 Create/Select Experiment', 'experiment'),
                    ('🚀 Train a new model', 'train'),
                    ('🧪 Test an existing model', 'test'),
                    ('📊 Benchmark multiple models', 'benchmark'),
                    ('⚙️  Configure settings', 'config'),
                    ('📖 View documentation', 'docs'),
                    ('❌ Exit', 'exit')
                ],
            ),
        ]
        
        answers = inquirer.prompt(questions)
        
        if answers['action'] == 'experiment':
            current_experiment = experiment_interactive()
        elif answers['action'] == 'train':
            if not current_experiment:
                console.print("[yellow]Please create/select an experiment first![/yellow]")
                console.input("\n[dim]Press Enter to continue...[/dim]")
            else:
                train_interactive(current_experiment)
        elif answers['action'] == 'test':
            if not current_experiment:
                console.print("[yellow]Please create/select an experiment first![/yellow]")
                console.input("\n[dim]Press Enter to continue...[/dim]")
            else:
                test_interactive(current_experiment)
        elif answers['action'] == 'benchmark':
            if not current_experiment:
                console.print("[yellow]Please create/select an experiment first![/yellow]")
                console.input("\n[dim]Press Enter to continue...[/dim]")
            else:
                benchmark_interactive(current_experiment)
        elif answers['action'] == 'config':
            config_interactive()
        elif answers['action'] == 'docs':
            show_documentation()
        elif answers['action'] == 'exit':
            console.print("\n[green]Goodbye! 👋[/green]\n")
            break


def train_interactive(experiment_config):
    """Interactive training setup"""
    console.clear()
    console.print(Panel.fit(
        f"[bold green]Training Setup[/bold green]\n"
        f"Experiment: [cyan]{experiment_config['experiment']['name']}[/cyan]",
        title="Configure Training"
    ))
    
    # Load available models from config
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.models_config import get_preprocessing_models, get_postprocessing_models, get_preprocessing_info, get_postprocessing_info
    
    preprocessing_models = get_preprocessing_models()
    postprocessing_models = get_postprocessing_models()
    
    # Create choices with descriptions for preprocessing
    prep_choices = []
    for model in preprocessing_models:
        info = get_preprocessing_info(model)
        prep_choices.append((f"{model} - {info['name']}", model))
    
    # Create choices with descriptions for postprocessing
    postp_choices = []
    for model in postprocessing_models:
        info = get_postprocessing_info(model)
        postp_choices.append((f"{info['name']} - {info['description']}", model))
    
    # Show dataset info from experiment
    console.print(f"\n[cyan]Using dataset from experiment:[/cyan]")
    console.print(f"  Train: [dim]{experiment_config['datasets']['train']}[/dim]")
    console.print(f"  ({experiment_config['datasets']['train_samples']} images)\n")
    
    # First, ask for model selection
    initial_questions = [
        inquirer.List(
            'preprocessing',
            message="Select preprocessing method",
            choices=prep_choices,
        ),
        inquirer.List(
            'postprocessing',
            message="Select post-processing model to train",
            choices=postp_choices + [('← Back to Main Menu', 'BACK')],
        ),
    ]
    
    initial_answers = inquirer.prompt(initial_questions)
    
    if not initial_answers or initial_answers.get('postprocessing') == 'BACK':
        return
    
    # Build the rest of the questions
    questions = []
    
    # Add UNet-specific parameters if UNet is selected
    if 'UNet' in initial_answers['postprocessing']:
        console.print("\n[cyan]UNet Architecture Configuration:[/cyan]")
        questions.extend([
            inquirer.Text(
                'num_encoders',
                message="Number of encoder-decoder blocks",
                default="3",
                validate=lambda _, x: x.isdigit() and 1 <= int(x) <= 5
            ),
            inquirer.Text(
                'start_middle_channels',
                message="Starting middle channels",
                default="64",
                validate=lambda _, x: x.isdigit() and int(x) > 0
            ),
        ])
    
    # Add common training parameters
    questions.extend([
        inquirer.Text(
            'epochs',
            message="Number of epochs",
            default="5",
            validate=lambda _, x: x.isdigit() and int(x) > 0
        ),
        inquirer.Text(
            'batch_size',
            message="Batch size",
            default="8",
            validate=lambda _, x: x.isdigit() and int(x) > 0
        ),
        inquirer.Text(
            'learning_rate',
            message="Learning rate",
            default="0.01"
        ),
        inquirer.List(
            'device',
            message="Select device",
            choices=[
                ('CUDA (GPU)', 'cuda'),
                ('CPU', 'cpu'),
            ],
        ),
    ])
    
    answers = inquirer.prompt(questions)
    
    if answers:
        # Combine initial answers with remaining answers
        answers.update(initial_answers)
        
        # Add dataset from experiment config to answers for summary display
        answers['dataset'] = experiment_config['datasets']['train']
        
        # Show summary
        show_training_summary(answers)
        
        if inquirer.confirm("Start training with these settings?", default=True):
            # Save training configuration automatically
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"{answers['preprocessing']}_{answers['postprocessing']}"
            config_name = f"training_{model_name}_{timestamp}.yaml"
            
            config_dir = Path(experiment_config['output_dirs']['logs'])
            config_dir.mkdir(parents=True, exist_ok=True)
            config_path = config_dir / config_name
            
            # Use dataset from experiment config
            train_dataset = experiment_config['datasets']['train']
            
            training_config = {
                'type': 'training',
                'model_name': model_name,
                'preprocessing': answers['preprocessing'],
                'postprocessing': answers['postprocessing'],
                'dataset': train_dataset,
                'epochs': int(answers['epochs']),
                'batch_size': int(answers['batch_size']),
                'learning_rate': float(answers['learning_rate']),
                'device': answers['device'],
                'started_at': datetime.now().isoformat(),
                'experiment': experiment_config['experiment']['name']
            }
            
            # Add UNet-specific parameters if present
            if 'num_encoders' in answers:
                training_config['num_encoders'] = int(answers['num_encoders'])
            if 'start_middle_channels' in answers:
                training_config['start_middle_channels'] = int(answers['start_middle_channels'])
            
            with open(config_path, 'w') as f:
                yaml.dump(training_config, f, default_flow_style=False, sort_keys=False)
            
            console.print(f"[dim]Configuration saved to: {config_path}[/dim]\n")
            
            # Start training
            from .commands import train_cmd
            output_dir = experiment_config['output_dirs']['models']
            
            # Prepare train_cmd arguments
            train_args = {
                'preprocessing': answers['preprocessing'],
                'postprocessing': answers['postprocessing'],
                'dataset': train_dataset,
                'epochs': int(answers['epochs']),
                'batch_size': int(answers['batch_size']),
                'lr': float(answers['learning_rate']),
                'output': output_dir
            }
            
            # Add UNet-specific parameters if present
            if 'num_encoders' in answers:
                train_args['num_encoders'] = int(answers['num_encoders'])
            if 'start_middle_channels' in answers:
                train_args['start_middle_channels'] = int(answers['start_middle_channels'])

            train_cmd(**train_args)
            
            console.input("\n[dim]Press Enter to continue...[/dim]")


def test_interactive(experiment_config):
    """Interactive testing setup"""
    console.clear()
    console.print(Panel.fit(
        f"[bold blue]Testing Setup[/bold blue]\n"
        f"Experiment: [cyan]{experiment_config['experiment']['name']}[/cyan]",
        title="Configure Testing"
    ))
    
    # List available models from experiment directory
    models_dir = Path(experiment_config['output_dirs']['models'])
    if models_dir.exists():
        available_models = list(models_dir.glob("*.pth"))
        if available_models:
            model_choices = [(f"{m.stem} ({m.stat().st_size / 1024:.1f} KB)", str(m)) 
                           for m in available_models]
        else:
            console.print(f"[yellow]No trained models found in {models_dir}.[/yellow]")
            console.print("[yellow]Please train a model first.[/yellow]")
            console.input("\n[dim]Press Enter to continue...[/dim]")
            return
    else:
        console.print(f"[yellow]Models directory not found: {models_dir}[/yellow]")
        console.input("\n[dim]Press Enter to continue...[/dim]")
        return
    
    # Show dataset info from experiment
    console.print(f"\n[cyan]Using test dataset from experiment:[/cyan]")
    console.print(f"  Test: [dim]{experiment_config['datasets']['test']}[/dim]")
    console.print(f"  ({experiment_config['datasets']['test_samples']} images)\n")
    
    questions = [
        inquirer.List(
            'checkpoint',
            message="Select model checkpoint",
            choices=model_choices + [('← Back to Main Menu', 'BACK')],
        ),
        
    ]
    
    answers = inquirer.prompt(questions)
    # Check if user wants to go back
    if not answers or answers.get('checkpoint') == 'BACK':
        return
    
    questions = [
        inquirer.Confirm(
            'visualize',
            message="Generate visualization plots?",
            default=True
        ),
    ]
    answers = inquirer.prompt(questions)
    # If visualization is enabled, ask how many samples
    if answers and answers['visualize']:
        num_samples_q = [
            inquirer.Text(
                'num_samples',
                message="How many samples to visualize? (default: 5)",
                default="5"
            )
        ]
        samples_answer = inquirer.prompt(num_samples_q)
        num_samples = int(samples_answer['num_samples']) if samples_answer else 5
    else:
        num_samples = 0
    
    if answers and inquirer.confirm("Start testing?", default=True):
        from .commands import test_cmd
        # Extract model name from checkpoint path
        model_name = Path(answers['checkpoint']).stem
        # Use test dataset from experiment config
        test_dataset = experiment_config['datasets']['test']
        test_cmd(
            model=model_name,
            checkpoint=answers['checkpoint'],
            dataset=test_dataset,
            output=experiment_config['output_dirs']['results'],
            experiment_name=experiment_config['experiment']['name'],
            visualize=answers['visualize'],
            num_samples=num_samples
        )
        
        console.input("\n[dim]Press Enter to continue...[/dim]")


def benchmark_interactive(experiment_config):
    """Interactive benchmark setup with preprocessing and postprocessing combinations"""
    console.clear()
    console.print(Panel.fit(
        f"[bold magenta]Benchmark Setup[/bold magenta]\n"
        f"Experiment: [cyan]{experiment_config['experiment']['name']}[/cyan]",
        title="Compare Model Combinations"
    ))
    
    # Load model configurations
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from utils.models_config import (get_preprocessing_models, get_postprocessing_models, 
                                     get_preprocessing_info, get_postprocessing_info,
                                     get_model_combinations)
    
    preprocessing_models = get_preprocessing_models()
    postprocessing_models = get_postprocessing_models()
    
    # Create choices for preprocessing
    prep_choices = []
    for model in preprocessing_models:
        info = get_preprocessing_info(model)
        prep_choices.append((f"{info['name']} - {info['description']}", model))
    
    # Create choices for postprocessing (multiple selection)
    postp_choices = []
    for model in postprocessing_models:
        info = get_postprocessing_info(model)
        postp_choices.append((f"{info['name']} - {info['description']}", model))
    
    # Show dataset info from experiment
    console.print(f"\n[cyan]Using test dataset from experiment:[/cyan]")
    console.print(f"  Test: [dim]{experiment_config['datasets']['test']}[/dim]")
    console.print(f"  ({experiment_config['datasets']['test_samples']} images)\n")
    
    # First ask if user wants to continue
    continue_question = [
        inquirer.List(
            'continue',
            message="Proceed with benchmark setup?",
            choices=[
                ('✓ Yes, configure benchmark', 'yes'),
                ('← Back to Main Menu', 'back')
            ]
        )
    ]
    
    continue_answer = inquirer.prompt(continue_question)
    if not continue_answer or continue_answer.get('continue') == 'back':
        return
    
    questions = [
        inquirer.Checkbox(
            'preprocessing',
            message="Select preprocessing methods to benchmark (Space to select, Enter to confirm)",
            choices=prep_choices,
            default=[preprocessing_models[0]] if preprocessing_models else []
        ),
        inquirer.Checkbox(
            'postprocessing',
            message="Select post-processing models to benchmark (Space to select, Enter to confirm)",
            choices=postp_choices,
        ),
    ]
    
    answers = inquirer.prompt(questions)
    
    if answers:
        prep_models = answers.get('preprocessing', [])
        postp_models = answers.get('postprocessing', [])
        
        if not prep_models:
            console.print("[yellow]Please select at least one preprocessing method.[/yellow]")
            console.input("\n[dim]Press Enter to continue...[/dim]")
            return
        
        if not postp_models:
            console.print("[yellow]Please select at least one post-processing model.[/yellow]")
            console.input("\n[dim]Press Enter to continue...[/dim]")
            return
        
        # Show summary
        console.print(f"\n[cyan]Benchmark Configuration:[/cyan]")
        console.print(f"  Preprocessing: {', '.join(prep_models)}")
        console.print(f"  Post-processing: {', '.join(postp_models)}")
        
        # Find actual trained models to show what will be tested
        import re
        from pathlib import Path
        models_dir = Path(experiment_config['output_dirs']['models'])
        
        if models_dir.exists():
            available_checkpoints = list(models_dir.glob("*.pth"))
            matching_models = []
            
            for checkpoint in available_checkpoints:
                checkpoint_name = checkpoint.stem
                for prep in prep_models:
                    for postp in postp_models:
                        pattern = f"^{re.escape(prep)}_{re.escape(postp)}(_.*)?$"
                        if re.match(pattern, checkpoint_name):
                            matching_models.append(checkpoint_name)
                            break
            
            if matching_models:
                console.print(f"\n[green]Found {len(matching_models)} trained model(s) to benchmark:[/green]")
                for model in matching_models:
                    console.print(f"  • {model}")
            else:
                console.print(f"\n[yellow]No trained models found matching the selected criteria.[/yellow]")
                console.print(f"[dim]Please train models first.[/dim]")
                console.input("\n[dim]Press Enter to continue...[/dim]")
                return
        else:
            console.print(f"\n[yellow]Trained models directory not found.[/yellow]")
            console.input("\n[dim]Press Enter to continue...[/dim]")
            return
        
        console.print()
        
        if inquirer.confirm(f"Run benchmark with these {len(matching_models)} model(s)?", default=True):
            from .commands import benchmark_cmd
            # Use test dataset from experiment config
            test_dataset = experiment_config['datasets']['test']
            benchmark_cmd(
                preprocessing=prep_models,
                postprocessing=postp_models,
                dataset=test_dataset,
                output=experiment_config['output_dirs']['benchmarks']
            )
            
            console.input("\n[dim]Press Enter to continue...[/dim]")


def config_interactive():
    """Interactive configuration"""
    console.clear()
    console.print(Panel.fit(
        "[bold yellow]Configuration[/bold yellow]",
        title="Settings"
    ))
    
    console.print("[dim]Configuration management coming soon...[/dim]")
    console.input("\n[dim]Press Enter to continue...[/dim]")


def show_documentation():
    """Show documentation"""
    console.clear()
    console.print(Panel.fit(
        "[bold cyan]Documentation[/bold cyan]",
        title="Help"
    ))
    
    console.print("""
[bold]Available Models:[/bold]
  • UNet_V1: U-Net architecture with skip connections
  • ThreeL_SSNet: Lightweight segmentation model

[bold]Workflow:[/bold]
  1. Train: Train a model on your dataset
  2. Test: Evaluate model performance
  3. Benchmark: Compare multiple models

[bold]Tips:[/bold]
  • Start with default parameters
  • Use GPU (CUDA) for faster training
  • Save configurations for reproducibility
    """)
    
    console.input("\n[dim]Press Enter to continue...[/dim]")


def show_training_summary(config):
    """Display training configuration summary"""
    table = Table(title="Training Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    
    # Handle both old format (model) and new format (preprocessing + postprocessing)
    if 'preprocessing' in config and 'postprocessing' in config:
        table.add_row("Preprocessing", config['preprocessing'])
        table.add_row("Post-processing", config['postprocessing'])
        table.add_row("Pipeline", f"{config['preprocessing']} → {config['postprocessing']}")
    elif 'model' in config:
        table.add_row("Model", config['model'])
    
    table.add_row("Dataset", config['dataset'])
    table.add_row("Epochs", config['epochs'])
    table.add_row("Batch Size", config['batch_size'])
    table.add_row("Learning Rate", config['learning_rate'])
    table.add_row("Device", config['device'])
    
    console.print("\n")
    console.print(table)
    console.print("\n")
