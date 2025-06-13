"""Console utilities for rich CLI output."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.markdown import Markdown
from rich.text import Text
from rich.align import Align

# Global console instance
console = Console()

def setup_console():
    """Setup the console with any necessary configuration."""
    return console

def create_header_panel(title: str, version: str = "v0.1.0", description: str = None) -> Panel:
    """Create a consistent header panel for CLI commands."""
    content = f"[bold blue]{title} {version}[/bold blue]"
    if description:
        content += f"\n{description}"
    
    return Panel.fit(content, border_style="blue")

def create_stats_table(title: str, data: dict, title_style: str = "cyan", value_style: str = "magenta") -> Table:
    """Create a statistics table with consistent styling."""
    table = Table(title=title)
    table.add_column("Metric", style=title_style)
    table.add_column("Value", style=value_style)
    
    for key, value in data.items():
        table.add_row(str(key), str(value))
    
    return table

def create_progress_context():
    """Create a standardized progress context."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    )

def create_detailed_progress_context():
    """Create a detailed progress context with bar and percentage."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    )

def print_success(message: str):
    """Print a success message with consistent styling."""
    console.print(f"[bold green]✓ {message}[/bold green]")

def print_error(message: str):
    """Print an error message with consistent styling."""
    console.print(f"[bold red]✗ {message}[/bold red]")

def print_warning(message: str):
    """Print a warning message with consistent styling."""
    console.print(f"[bold yellow]⚠ {message}[/bold yellow]")

def print_info(message: str):
    """Print an info message with consistent styling."""
    console.print(f"[bold blue]ℹ {message}[/bold blue]")

def print_section_header(title: str):
    """Print a section header with consistent styling."""
    console.print(f"\n[bold cyan]═══ {title} ═══[/bold cyan]")

def print_subsection_header(title: str):
    """Print a subsection header with consistent styling."""
    console.print(f"\n[bold blue]─── {title} ───[/bold blue]")

def create_confirmation_panel(message: str, is_destructive: bool = False) -> Panel:
    """Create a confirmation panel for user prompts."""
    style = "red" if is_destructive else "yellow"
    return Panel(
        message,
        title="[bold]Confirmation Required[/bold]",
        border_style=style
    )

def format_count(count: int, singular: str, plural: str = None) -> str:
    """Format a count with proper singular/plural form."""
    if plural is None:
        plural = f"{singular}s"
    
    if count == 1:
        return f"1 {singular}"
    else:
        return f"{count} {plural}"

def create_tag_display(tags: list, max_display: int = 10) -> str:
    """Create a formatted display of tags."""
    if not tags:
        return "[dim]No tags[/dim]"
    
    if len(tags) <= max_display:
        return " ".join([f"[blue]#{tag}[/blue]" for tag in tags])
    else:
        displayed = tags[:max_display]
        remaining = len(tags) - max_display
        tag_display = " ".join([f"[blue]#{tag}[/blue]" for tag in displayed])
        return f"{tag_display} [dim]+{remaining} more[/dim]"

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to a maximum length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix