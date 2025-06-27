# PM Automation Suite - Quick Start Guide

Get up and running with PM Automation Suite in 5 minutes!

## 🚀 Installation

```bash
# From the obsidian-vault-tools directory
cd pm_automation_suite
pip install -e ".[all]"
```

## 🔑 Essential Configuration

1. **Create `.env` file**:
```bash
cp .env.example .env
```

2. **Add minimum credentials**:
```env
# Required
OBSIDIAN_VAULT_PATH=/path/to/your/vault
OPENAI_API_KEY=sk-...

# For Jira integration
JIRA_URL=https://company.atlassian.net
JIRA_EMAIL=you@company.com
JIRA_API_TOKEN=your-token

# For Google integration (optional)
GOOGLE_CLIENT_ID=your-client-id
GOOGLE_CLIENT_SECRET=your-secret
```

## 🎯 Quick Wins

### 1. Generate WBR in 2 Minutes

```bash
# Launch the unified manager
ovt

# Navigate: PM Tools → WBR/QBR Automation → Quick WBR
# Select: Last Week's Data
# Click: Generate Report
```

### 2. Convert PRD to Jira Stories

```bash
# In the unified manager
# Navigate: PM Tools → Feature Development Pipeline
# Select: Parse PRD Document
# Choose your PRD file
# Review and create stories
```

### 3. Check Team Health

```bash
# Navigate: PM Tools → Analytics Hub
# Select: Team Health Dashboard
# View burnout risk scores
```

## 📊 Common Workflows

### Weekly Review Automation

```python
from pm_automation_suite.wbr import WBROrchestrator

# This runs automatically when configured through UI
orchestrator = WBROrchestrator()
await orchestrator.generate_weekly_review(
    project="PROD",
    output_format="slides"
)
```

### Bulk Story Creation

```python
from pm_automation_suite.feature_dev import FeaturePipeline

# Parse PRD and create stories
pipeline = FeaturePipeline()
stories = await pipeline.prd_to_stories(
    prd_file="requirements.pdf",
    project_key="FEAT"
)
```

## 🛠️ Troubleshooting

### Issue: "No module named 'pm_automation_suite'"
```bash
# Ensure you're in the correct directory
cd /path/to/obsidian-vault-tools
pip install -e .
```

### Issue: "Authentication failed"
```bash
# Re-authenticate through UI
ovt
# Navigate: PM Tools → PM Suite Configuration
# Select: Re-authenticate [Service]
```

### Issue: "AI generation timeout"
```bash
# Check your OpenAI API key
echo $OPENAI_API_KEY
# Verify it starts with 'sk-'
```

## 📚 Next Steps

1. Read the full [User Guide](USER_GUIDE.md)
2. Explore [example workflows](../examples/)
3. Join the community discussions
4. Customize for your team's needs

## 💡 Pro Tips

1. **Start with one integration** - Don't try to automate everything at once
2. **Use templates** - Build on provided templates before creating custom ones
3. **Monitor logs** - Check `~/.pm_automation_suite/logs/` for detailed info
4. **Test first** - Run workflows manually before scheduling
5. **Iterate** - Refine AI prompts based on output quality

## 🔗 Useful Commands

```bash
# Check status
ovt
# PM Tools → PM Suite Configuration → System Status

# View logs
tail -f ~/.pm_automation_suite/logs/automation.log

# Run tests
cd pm_automation_suite
pytest tests/

# Start API server (for integrations)
uvicorn pm_automation_suite.api.main:app --reload
```

## 🎉 You're Ready!

Launch `ovt` and start automating your PM workflows. Remember: the goal is to save time on repetitive tasks so you can focus on strategy and innovation.

Need help? Check the [User Guide](USER_GUIDE.md) or file an issue on GitHub!