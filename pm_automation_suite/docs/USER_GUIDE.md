# PM Automation Suite User Guide

Welcome to the PM Automation Suite! This guide will help you get started with automating your product management workflows.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Authentication Setup](#authentication-setup)
3. [WBR/QBR Automation](#wbrqbr-automation)
4. [Feature Development Pipeline](#feature-development-pipeline)
5. [Analytics Hub](#analytics-hub)
6. [Real-time Monitoring](#real-time-monitoring)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Getting Started

### Prerequisites

Before using the PM Automation Suite, ensure you have:

1. **Obsidian Vault Tools installed** (v2.3.0 or higher)
2. **API credentials** for the services you want to integrate:
   - Jira API token
   - Google OAuth credentials (for Google Workspace)
   - Snowflake connection details (optional)
   - OpenAI API key (for AI features)

### Initial Setup

1. **Launch the unified manager**:
   ```bash
   ovt
   ```

2. **Navigate to PM Tools** → **PM Suite Configuration**

3. **Configure your integrations** following the prompts

## Authentication Setup

The PM Automation Suite uses OAuth 2.0 for secure authentication with external services.

### Google Workspace Setup

1. Create OAuth credentials in Google Cloud Console:
   - Go to [Google Cloud Console](https://console.cloud.google.com)
   - Create a new project or select existing
   - Enable necessary APIs (Sheets, Slides, Drive)
   - Create OAuth 2.0 credentials
   - Add authorized redirect URI: `http://localhost:8080`

2. In the PM Suite Configuration:
   ```
   Select: Configure Google OAuth
   Enter Client ID: [your-client-id]
   Enter Client Secret: [your-client-secret]
   ```

3. A browser window will open for authorization
4. Grant the requested permissions
5. The suite will store your tokens securely

### Jira Setup

1. Generate an API token:
   - Log into Jira
   - Go to Account Settings → Security → API tokens
   - Create new token

2. Configure in PM Suite:
   ```
   Select: Configure Jira Connection
   Enter Jira URL: https://your-company.atlassian.net
   Enter Email: your-email@company.com
   Enter API Token: [your-token]
   ```

## WBR/QBR Automation

The WBR/QBR automation helps you generate weekly and quarterly business reviews automatically.

### Setting Up Your First WBR

1. **Navigate to**: PM Tools → WBR/QBR Automation

2. **Choose data sources**:
   - Jira: Sprint metrics, velocity, issues
   - Google Sheets: Custom metrics, KPIs
   - Snowflake: Business metrics (if configured)

3. **Configure the workflow**:
   ```
   Select: Create New WBR Workflow
   Name: Weekly Product Review
   Frequency: Weekly (Mondays at 9 AM)
   Data Sources: [Select your sources]
   ```

4. **Customize insights**:
   - The AI will analyze trends automatically
   - You can provide specific focus areas
   - Custom prompts for deeper analysis

5. **Generate presentation**:
   - Choose from templates or create custom
   - AI generates slides with insights
   - Review and edit before distribution

### Example WBR Workflow

```python
# This happens automatically when you configure through the UI
workflow = {
    "name": "Weekly Product Review",
    "schedule": "0 9 * * 1",  # Every Monday at 9 AM
    "data_sources": [
        {
            "type": "jira",
            "config": {
                "jql": "project = PROD AND updated >= -1w",
                "fields": ["status", "priority", "story_points"]
            }
        },
        {
            "type": "google_sheets",
            "config": {
                "spreadsheet_id": "your-metrics-sheet-id",
                "range": "Weekly!A:F"
            }
        }
    ],
    "insights": {
        "focus_areas": ["velocity", "blockers", "upcoming_risks"],
        "ai_model": "gpt-4"
    },
    "output": {
        "format": "powerpoint",
        "template": "executive_summary",
        "distribution": ["team@company.com"]
    }
}
```

## Feature Development Pipeline

Automate the journey from PRD to Jira stories with AI assistance.

### PRD to Stories Workflow

1. **Upload your PRD**:
   - Navigate to: PM Tools → Feature Development Pipeline
   - Select: Parse PRD Document
   - Choose your PRD file (PDF, DOCX, or Markdown)

2. **Review extracted requirements**:
   - The system extracts key requirements
   - AI identifies user personas and use cases
   - Review and adjust as needed

3. **Generate user stories**:
   - AI creates stories with acceptance criteria
   - Estimates story points based on complexity
   - Generates Gherkin scenarios for testing

4. **Bulk create in Jira**:
   - Review generated stories
   - Map to appropriate epic
   - Set priorities and assignments
   - Create all stories with one click

### Story Generation Example

Input PRD excerpt:
```
The system should allow users to export their data in multiple formats including CSV, JSON, and PDF. The export should be asynchronous for large datasets.
```

Generated stories:
```
Title: Export data in multiple formats
As a: Power User
I want to: Export my data in CSV, JSON, or PDF format
So that: I can analyze data in external tools

Acceptance Criteria:
- GIVEN I have data to export
  WHEN I select CSV format
  THEN the system generates a valid CSV file

- GIVEN I have more than 10,000 records
  WHEN I request an export
  THEN the system processes it asynchronously
  AND notifies me when complete

Story Points: 5
Priority: High
```

## Analytics Hub

Track PM performance metrics and predict trends using ML models.

### Setting Up Analytics

1. **Configure data sources**:
   - PM Tools → Analytics Hub & ML Insights
   - Connect your data warehouse (optional)
   - Set up ETL pipelines for regular updates

2. **Choose metrics to track**:
   - Sprint velocity trends
   - Feature adoption rates
   - Team productivity metrics
   - Customer satisfaction scores

3. **Enable ML predictions**:
   - Burnout risk detection
   - Sprint completion probability
   - Feature success prediction
   - Resource allocation optimization

### Dashboard Creation

1. **Select dashboard type**:
   - Executive Summary
   - Team Performance
   - Product Health
   - Custom Dashboard

2. **Configure visualizations**:
   ```
   Metric: Sprint Velocity
   Type: Line Chart
   Period: Last 6 Sprints
   Add Trend Line: Yes
   Predictions: Next 2 Sprints
   ```

3. **Set up alerts**:
   - Velocity dropping below threshold
   - Burnout risk above 70%
   - Sprint goals at risk

### ML Model Examples

**Burnout Detection**:
- Analyzes: Work hours, ticket volume, communication patterns
- Predicts: Risk level (Low/Medium/High)
- Recommends: Workload adjustments, time off

**Sprint Success Prediction**:
- Analyzes: Historical velocity, team composition, scope
- Predicts: Completion probability
- Recommends: Scope adjustments, resource allocation

## Real-time Monitoring

Monitor your PM metrics in real-time with customizable alerts.

### Setting Up Monitoring

1. **Configure metrics**:
   ```
   PM Tools → Real-time Monitoring
   Select: Configure Metrics
   
   Add Metric:
   - Name: Active Sprint Progress
   - Source: Jira
   - Query: Issues in current sprint
   - Threshold: < 50% complete by mid-sprint
   - Alert: Email + Slack
   ```

2. **Create alert rules**:
   - Metric thresholds
   - Anomaly detection
   - Trend alerts
   - Custom conditions

3. **Integration options**:
   - Prometheus (for existing monitoring)
   - Webhook notifications
   - Email alerts
   - Slack integration

### Monitoring Dashboard

The real-time dashboard shows:
- Current sprint burndown
- Team velocity trends
- Blocker count and age
- Resource utilization
- Custom metrics

## Best Practices

### 1. Start Small
- Begin with one workflow (e.g., WBR automation)
- Test with a small dataset
- Gradually add more sources and complexity

### 2. Data Quality
- Ensure Jira hygiene (proper story points, status updates)
- Maintain consistent Google Sheets formats
- Regular data validation

### 3. AI Optimization
- Provide clear context in prompts
- Review and refine AI outputs
- Build a library of successful prompts

### 4. Security
- Rotate API tokens regularly
- Use OAuth where possible
- Review access permissions quarterly
- Never share credentials

### 5. Workflow Design
- Keep workflows focused and simple
- Test thoroughly before scheduling
- Monitor execution logs
- Have fallback plans

## Troubleshooting

### Common Issues

**Authentication Errors**:
```
Error: Invalid credentials
Solution: 
1. Check API token hasn't expired
2. Verify correct URL/email
3. Re-authenticate through PM Suite Configuration
```

**Data Extraction Failures**:
```
Error: No data returned from Jira
Solution:
1. Verify JQL query syntax
2. Check project permissions
3. Ensure fields exist in Jira
```

**AI Generation Issues**:
```
Error: AI response timeout
Solution:
1. Reduce data volume
2. Simplify prompts
3. Check API rate limits
4. Verify API key validity
```

### Debug Mode

Enable debug logging for detailed troubleshooting:

1. In PM Suite Configuration:
   ```
   Select: Enable Debug Mode
   ```

2. Or set environment variable:
   ```bash
   export PM_SUITE_DEBUG=true
   ```

3. Check logs at:
   ```
   ~/.pm_automation_suite/logs/debug.log
   ```

### Getting Help

1. **Check the logs**: Most errors include helpful messages
2. **Documentation**: Refer to the README for technical details
3. **Community**: Ask in the Obsidian Vault Tools discussions
4. **Issues**: Report bugs on GitHub

## Advanced Usage

### Custom Workflows

Create complex multi-step workflows:

```python
# Example: End-to-end feature workflow
custom_workflow = {
    "name": "Feature Launch Pipeline",
    "steps": [
        {
            "action": "parse_prd",
            "input": "features/new-feature.md"
        },
        {
            "action": "generate_stories",
            "config": {
                "include_test_cases": true,
                "estimate_points": true
            }
        },
        {
            "action": "create_jira_epic",
            "config": {
                "project": "FEAT",
                "epic_name": "{{feature_name}}"
            }
        },
        {
            "action": "schedule_reviews",
            "config": {
                "frequency": "weekly",
                "stakeholders": ["pm@company.com", "eng@company.com"]
            }
        }
    ]
}
```

### API Integration

Use the PM Automation Suite API for custom integrations:

```bash
# Start the API server
cd pm_automation_suite
uvicorn api.main:app --reload

# Example API calls
# Generate insights
curl -X POST http://localhost:8000/api/insights \
  -H "Content-Type: application/json" \
  -d '{"data_source": "jira", "project": "PROD"}'

# Create workflow
curl -X POST http://localhost:8000/api/workflows \
  -H "Content-Type: application/json" \
  -d @workflow.json
```

### Extending the Suite

Add custom connectors or processors:

1. Create new connector in `connectors/`
2. Implement `BaseConnector` interface
3. Register in configuration
4. Use in workflows

## Conclusion

The PM Automation Suite is designed to save you time and provide better insights into your product management processes. Start with the features that provide the most immediate value, and gradually expand your automation as you become comfortable with the system.

Remember: The goal is to augment your PM skills, not replace them. Use the automation to handle repetitive tasks and data gathering, freeing you to focus on strategy and decision-making.

Happy automating! 🚀