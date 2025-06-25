#!/usr/bin/env python3
"""
PM Daily Template Generator - Automated burnout prevention daily note creation.

This module generates enhanced daily notes for Product Managers with:
- Auto-populated WSJF priorities
- Rotating product focus based on day of week
- Completion rate tracking
- Energy and burnout monitoring
"""

import os
import re
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PMDailyTemplateGenerator:
    """Generator for enhanced PM daily notes with burnout prevention features."""
    
    def __init__(self, vault_path: str):
        """
        Initialize the generator with vault path.
        
        Args:
            vault_path: Path to the Obsidian vault
        """
        self.vault_path = Path(vault_path)
        self.daily_notes_path = self.vault_path / "Daily Notes"
        self.templates_path = self.vault_path / "Templates"
        self.template_path = self.templates_path / "PM_Daily_Template_Enhanced.md"
        
        # Product rotation schedule
        self.product_schedule = {
            0: "DFP 2.0",  # Monday
            1: "DFP 2.0",  # Tuesday
            2: "Payment Protection",  # Wednesday
            3: "Payment Protection",  # Thursday
            4: "Identity Intelligence",  # Friday
            5: "Review & Planning",  # Saturday
            6: "Review & Planning"   # Sunday
        }
        
    def read_wsjf_priorities(self) -> List[Dict[str, any]]:
        """
        Read WSJF priorities from existing analysis files.
        
        Returns:
            List of top 3 WSJF tasks with scores
        """
        wsjf_file = self.vault_path / "PM_BURNOUT_SOLUTION_TRACKER.md"
        tasks = []
        
        if wsjf_file.exists():
            try:
                with open(wsjf_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Extract WSJF tasks using regex
                # Look for pattern: [WSJF: score] Task description
                pattern = r'\[WSJF:\s*(\d+(?:\.\d+)?)\]\s*(.+?)(?:\n|$)'
                matches = re.findall(pattern, content)
                
                for score, task in matches:
                    tasks.append({
                        'score': float(score),
                        'task': task.strip()
                    })
                
                # Sort by WSJF score descending and return top 3
                tasks.sort(key=lambda x: x['score'], reverse=True)
                return tasks[:3]
                
            except Exception as e:
                logger.error(f"Error reading WSJF priorities: {e}")
                
        return []
    
    def calculate_completion_rates(self, days_back: int = 7) -> Tuple[float, float]:
        """
        Calculate completion rates from previous daily notes.
        
        Args:
            days_back: Number of days to look back for average
            
        Returns:
            Tuple of (yesterday's rate, week average rate)
        """
        yesterday_rate = 0.0
        week_rates = []
        
        for i in range(1, days_back + 1):
            date = datetime.now() - timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            daily_note = self.daily_notes_path / f"{date_str}.md"
            
            if daily_note.exists():
                try:
                    with open(daily_note, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Extract completion rate
                    rate_match = re.search(r'\*\*Completion Rate\*\*:\s*(\d+)%', content)
                    if rate_match:
                        rate = float(rate_match.group(1))
                        week_rates.append(rate)
                        
                        if i == 1:  # Yesterday
                            yesterday_rate = rate
                            
                except Exception as e:
                    logger.error(f"Error reading daily note {date_str}: {e}")
        
        week_average = sum(week_rates) / len(week_rates) if week_rates else 0.0
        
        return yesterday_rate, week_average
    
    def get_product_focus(self, date: Optional[datetime] = None) -> str:
        """
        Get the product focus for a given date based on rotation schedule.
        
        Args:
            date: Date to get focus for (defaults to today)
            
        Returns:
            Product name for focus
        """
        if date is None:
            date = datetime.now()
            
        weekday = date.weekday()
        return self.product_schedule.get(weekday, "DFP 2.0")
    
    def generate_daily_note(self, date: Optional[datetime] = None) -> str:
        """
        Generate enhanced daily note content.
        
        Args:
            date: Date to generate note for (defaults to today)
            
        Returns:
            Generated daily note content
        """
        if date is None:
            date = datetime.now()
            
        # Read template
        if not self.template_path.exists():
            raise FileNotFoundError(f"Template not found: {self.template_path}")
            
        with open(self.template_path, 'r', encoding='utf-8') as f:
            template = f.read()
            
        # Get data
        wsjf_tasks = self.read_wsjf_priorities()
        product_focus = self.get_product_focus(date)
        yesterday_rate, week_average = self.calculate_completion_rates()
        
        # Replace template variables
        replacements = {
            '<% tp.date.now("YYYY-MM-DD") %>': date.strftime("%Y-%m-%d"),
            '<% tp.date.now("dddd") %>': date.strftime("%A"),
            '<% tp.date.now("GGGG-[W]WW") %>': date.strftime("%G-W%V"),
            '<% tp.date.now("dddd, MMMM D, YYYY") %>': date.strftime("%A, %B %-d, %Y"),
        }
        
        content = template
        for old, new in replacements.items():
            content = content.replace(old, new)
            
        # Update WSJF priorities
        if wsjf_tasks:
            for i, task in enumerate(wsjf_tasks[:3]):
                old_line = f"{i+1}. [ ] [WSJF: ] Task: "
                new_line = f"{i+1}. [ ] [WSJF: {task['score']:.1f}] {task['task']}"
                content = content.replace(old_line, new_line)
                
        # Update product focus
        focus_checkboxes = {
            "DFP 2.0": "[x] DFP 2.0 | [ ] Payment Protection | [ ] Identity Intelligence",
            "Payment Protection": "[ ] DFP 2.0 | [x] Payment Protection | [ ] Identity Intelligence",
            "Identity Intelligence": "[ ] DFP 2.0 | [ ] Payment Protection | [x] Identity Intelligence",
            "Review & Planning": "[ ] DFP 2.0 | [ ] Payment Protection | [ ] Identity Intelligence"
        }
        
        old_focus = "[ ] DFP 2.0 | [ ] Payment Protection | [ ] Identity Intelligence"
        new_focus = focus_checkboxes.get(product_focus, old_focus)
        content = content.replace(f"**Today's Focus**: {old_focus}", 
                                f"**Today's Focus**: {new_focus}")
        
        # Update completion rates
        if week_average > 0:
            content = content.replace("**Week Average**: %", 
                                    f"**Week Average**: {week_average:.0f}%")
            
        return content
    
    def create_daily_note(self, date: Optional[datetime] = None) -> Path:
        """
        Create a new daily note file.
        
        Args:
            date: Date to create note for (defaults to today)
            
        Returns:
            Path to created daily note
        """
        if date is None:
            date = datetime.now()
            
        # Ensure daily notes directory exists
        self.daily_notes_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        filename = f"{date.strftime('%Y-%m-%d')}.md"
        filepath = self.daily_notes_path / filename
        
        # Check if file already exists
        if filepath.exists():
            logger.warning(f"Daily note already exists: {filepath}")
            return filepath
            
        # Generate content
        content = self.generate_daily_note(date)
        
        # Write file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
            
        logger.info(f"Created daily note: {filepath}")
        return filepath
    
    def update_completion_rate(self, date: Optional[datetime] = None,
                             tasks_completed: int = 0) -> None:
        """
        Update completion rate in existing daily note.
        
        Args:
            date: Date of note to update (defaults to today)
            tasks_completed: Number of tasks completed
        """
        if date is None:
            date = datetime.now()
            
        filename = f"{date.strftime('%Y-%m-%d')}.md"
        filepath = self.daily_notes_path / filename
        
        if not filepath.exists():
            logger.error(f"Daily note not found: {filepath}")
            return
            
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Calculate completion rate (3 tasks planned by default)
        rate = (tasks_completed / 3) * 100 if tasks_completed <= 3 else 100
        
        # Update metrics
        content = re.sub(r'\*\*Tasks Completed\*\*:\s*\n',
                        f'**Tasks Completed**: {tasks_completed}\n', content)
        content = re.sub(r'\*\*Completion Rate\*\*:\s*%',
                        f'**Completion Rate**: {rate:.0f}%', content)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
            
        logger.info(f"Updated completion rate for {date.strftime('%Y-%m-%d')}: {rate:.0f}%")


def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate enhanced PM daily notes with burnout prevention"
    )
    parser.add_argument(
        "--vault-path",
        type=str,
        default=os.environ.get("OBSIDIAN_VAULT_PATH", "/Users/nvaldez/Documents/Obsidian Vault"),
        help="Path to Obsidian vault"
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Date to generate note for (YYYY-MM-DD format, defaults to today)"
    )
    parser.add_argument(
        "--update-completion",
        type=int,
        metavar="TASKS",
        help="Update completion rate with number of tasks completed"
    )
    
    args = parser.parse_args()
    
    # Parse date if provided
    date = None
    if args.date:
        try:
            date = datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            logger.error(f"Invalid date format: {args.date}. Use YYYY-MM-DD")
            return
            
    # Create generator
    generator = PMDailyTemplateGenerator(args.vault_path)
    
    # Update completion or create new note
    if args.update_completion is not None:
        generator.update_completion_rate(date, args.update_completion)
    else:
        filepath = generator.create_daily_note(date)
        print(f"Created daily note: {filepath}")


if __name__ == "__main__":
    main()