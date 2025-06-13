#!/usr/bin/env python3
"""Entry point for the Obsidian Librarian CLI."""

def main():
    """Main entry point for the CLI."""
    try:
        from .cli.main_typer import app
        app()
    except ImportError as e:
        # Fallback to old CLI if new one fails
        print(f"Warning: New CLI not available ({e}), falling back to old CLI")
        try:
            from .cli.main import app
            app()
        except ImportError:
            from .cli import cli
            cli()

# Export the main function for script entry point
app = main

if __name__ == "__main__":
    main()