"""PM Tools menu integration for Unified Vault Manager."""

from typing import Optional
from datetime import datetime
import logging

from ..utils import Colors
from .daily_template_generator import PMDailyTemplateGenerator

logger = logging.getLogger(__name__)


class PMToolsMenu:
    """PM Tools menu handler for Unified Vault Manager."""
    
    def __init__(self, vault_path: str):
        """Initialize PM Tools menu."""
        self.vault_path = vault_path
        self.generator = PMDailyTemplateGenerator(vault_path)
        
    def display_menu(self) -> Optional[str]:
        """Display PM Tools menu and get user choice."""
        print(f"\n{Colors.HEADER}=== PM Tools Menu ==={Colors.ENDC}")
        print(f"{Colors.OKGREEN}1.{Colors.ENDC} Generate Today's Enhanced Daily Note")
        print(f"{Colors.OKGREEN}2.{Colors.ENDC} Generate Daily Note for Specific Date")
        print(f"{Colors.OKGREEN}3.{Colors.ENDC} Update Today's Completion Rate")
        print(f"{Colors.OKGREEN}4.{Colors.ENDC} View This Week's Product Focus Schedule")
        print(f"{Colors.OKGREEN}5.{Colors.ENDC} View Weekly Completion Stats")
        print(f"{Colors.WARNING}0.{Colors.ENDC} Back to Main Menu")
        
        return input(f"\n{Colors.OKBLUE}Select option: {Colors.ENDC}")
        
    def run(self) -> bool:
        """
        Run the PM Tools menu.
        
        Returns:
            True to continue in main menu, False to exit
        """
        while True:
            choice = self.display_menu()
            
            if choice == "0":
                return True
                
            elif choice == "1":
                self._generate_today_note()
                
            elif choice == "2":
                self._generate_specific_date_note()
                
            elif choice == "3":
                self._update_completion_rate()
                
            elif choice == "4":
                self._view_product_schedule()
                
            elif choice == "5":
                self._view_weekly_stats()
                
            else:
                print(f"{Colors.FAIL}Invalid option. Please try again.{Colors.ENDC}")
                
    def _generate_today_note(self):
        """Generate today's enhanced daily note."""
        try:
            filepath = self.generator.create_daily_note()
            print(f"{Colors.OKGREEN}✓ Created daily note: {filepath}{Colors.ENDC}")
            
            # Show product focus for today
            focus = self.generator.get_product_focus()
            print(f"{Colors.OKBLUE}Today's Product Focus: {focus}{Colors.ENDC}")
            
        except FileExistsError:
            print(f"{Colors.WARNING}Daily note for today already exists{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.FAIL}Error creating daily note: {e}{Colors.ENDC}")
            
    def _generate_specific_date_note(self):
        """Generate daily note for a specific date."""
        date_str = input(f"{Colors.OKBLUE}Enter date (YYYY-MM-DD): {Colors.ENDC}")
        
        try:
            date = datetime.strptime(date_str, "%Y-%m-%d")
            filepath = self.generator.create_daily_note(date)
            print(f"{Colors.OKGREEN}✓ Created daily note: {filepath}{Colors.ENDC}")
            
            # Show product focus for that date
            focus = self.generator.get_product_focus(date)
            print(f"{Colors.OKBLUE}Product Focus for {date_str}: {focus}{Colors.ENDC}")
            
        except ValueError:
            print(f"{Colors.FAIL}Invalid date format. Use YYYY-MM-DD{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.FAIL}Error creating daily note: {e}{Colors.ENDC}")
            
    def _update_completion_rate(self):
        """Update today's completion rate."""
        try:
            tasks_str = input(f"{Colors.OKBLUE}Number of tasks completed (out of 3): {Colors.ENDC}")
            tasks_completed = int(tasks_str)
            
            if tasks_completed < 0:
                print(f"{Colors.FAIL}Number cannot be negative{Colors.ENDC}")
                return
                
            self.generator.update_completion_rate(tasks_completed=tasks_completed)
            
            rate = min((tasks_completed / 3) * 100, 100)
            print(f"{Colors.OKGREEN}✓ Updated completion rate: {rate:.0f}%{Colors.ENDC}")
            
        except ValueError:
            print(f"{Colors.FAIL}Please enter a valid number{Colors.ENDC}")
        except Exception as e:
            print(f"{Colors.FAIL}Error updating completion rate: {e}{Colors.ENDC}")
            
    def _view_product_schedule(self):
        """View this week's product focus schedule."""
        print(f"\n{Colors.HEADER}=== Weekly Product Focus Schedule ==={Colors.ENDC}")
        
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        for i, day in enumerate(days):
            focus = self.generator.product_schedule[i]
            today_marker = " ← TODAY" if i == datetime.now().weekday() else ""
            print(f"{Colors.OKGREEN}{day}:{Colors.ENDC} {focus}{today_marker}")
            
    def _view_weekly_stats(self):
        """View weekly completion statistics."""
        yesterday_rate, week_average = self.generator.calculate_completion_rates()
        
        print(f"\n{Colors.HEADER}=== Weekly Completion Stats ==={Colors.ENDC}")
        print(f"{Colors.OKBLUE}Yesterday's Rate:{Colors.ENDC} {yesterday_rate:.0f}%")
        print(f"{Colors.OKBLUE}7-Day Average:{Colors.ENDC} {week_average:.0f}%")
        
        # Show trend
        if yesterday_rate > week_average:
            print(f"{Colors.OKGREEN}↑ Trending up!{Colors.ENDC}")
        elif yesterday_rate < week_average:
            print(f"{Colors.WARNING}↓ Trending down{Colors.ENDC}")
        else:
            print(f"{Colors.OKBLUE}→ Stable{Colors.ENDC}")