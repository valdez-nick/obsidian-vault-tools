# PM Tools for Obsidian Vault

Product Manager tools for burnout prevention and productivity optimization.

## Features

### Enhanced Daily Template Generator

The `daily_template_generator.py` module provides automated generation of PM-focused daily notes with:

- **Auto-populated WSJF Priorities**: Pulls top 3 tasks from your WSJF analysis
- **Smart Product Rotation**: Automatically assigns daily product focus based on schedule
- **Completion Tracking**: Calculates daily and weekly completion rates
- **Energy Monitoring**: Track energy levels and burnout indicators
- **Burnout Prevention**: Enforces limits (max 3 tasks) and tracks context switches

### Product Focus Schedule

Default rotation to minimize context switching:
- **Monday/Tuesday**: DFP 2.0
- **Wednesday/Thursday**: Payment Protection
- **Friday**: Identity Intelligence
- **Weekend**: Review & Planning

## Usage

### Command Line

```bash
# Generate today's enhanced daily note
python -m obsidian_vault_tools.pm_tools.daily_template_generator

# Generate for specific date
python -m obsidian_vault_tools.pm_tools.daily_template_generator --date 2025-01-30

# Update completion rate
python -m obsidian_vault_tools.pm_tools.daily_template_generator --update-completion 2
```

### Via Unified Vault Manager

The PM Tools are integrated into the main OVT menu system. Simply run `ovt` and navigate to the PM Tools section.

### Python API

```python
from obsidian_vault_tools.pm_tools.daily_template_generator import PMDailyTemplateGenerator

# Initialize generator
generator = PMDailyTemplateGenerator("/path/to/vault")

# Create today's note
filepath = generator.create_daily_note()

# Update completion rate
generator.update_completion_rate(tasks_completed=2)

# Get product focus for any date
from datetime import datetime
date = datetime(2025, 1, 30)
focus = generator.get_product_focus(date)
```

## Template Structure

The enhanced template includes all standard daily note sections plus:

1. **WSJF Priorities Section**: Top 3 tasks with scores
2. **Product Focus Section**: Daily product assignment with context switch tracking
3. **Energy Monitoring Section**: AM/PM energy levels and burnout indicators
4. **Daily Metrics Section**: Completion rates and weekly averages

## Configuration

### WSJF Task Source

Tasks are read from `PM_BURNOUT_SOLUTION_TRACKER.md` in your vault root. Format:
```markdown
- [WSJF: 35.5] Task description here
```

### Customizing Product Schedule

Edit the `product_schedule` dictionary in `daily_template_generator.py`:
```python
self.product_schedule = {
    0: "Your Monday Product",
    1: "Your Tuesday Product",
    # etc...
}
```

## Burnout Prevention Best Practices

1. **Enforce Top 3 Rule**: Never add more than 3 high-priority tasks
2. **Single Product Focus**: Stick to the assigned product for the day
3. **Track Energy**: Update energy levels twice daily
4. **Complete Deep Work Blocks**: Protect 9-12 AM and 2-5 PM
5. **Monitor Trends**: Review weekly averages to spot burnout early

## File Locations

- **Templates**: `{vault}/Templates/PM_Daily_Template_Enhanced.md`
- **Daily Notes**: `{vault}/Daily Notes/YYYY-MM-DD.md`
- **WSJF Tracker**: `{vault}/PM_BURNOUT_SOLUTION_TRACKER.md`

## Requirements

- Python 3.8+
- Obsidian vault with standard folder structure
- Templater plugin (for template variables)
- Meta Bind plugin (for interactive buttons)