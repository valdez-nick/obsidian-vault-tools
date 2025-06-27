# PM Automation Suite

A comprehensive automation suite for Product Management workflows, integrating with popular PM tools and leveraging AI for intelligent insights and content generation.

## 🚀 Features

### Core Capabilities
- **Multi-Source Data Integration**: Connect to Jira, Confluence, Google Suite, and Snowflake
- **OAuth 2.0 Authentication**: Secure OAuth flows for Google, Atlassian, and Microsoft with automatic token refresh
- **Multi-Tenant Support**: Manage credentials and connections for multiple workspaces/tenants
- **AI-Powered Intelligence**: Automated analysis, content generation, and insights using OpenAI and Anthropic
- **Workflow Orchestration**: Event-driven automation with scheduling capabilities
- **Flexible Architecture**: Modular design for easy extension and customization
- **Security First**: Built-in credential management with keyring integration and encryption at rest

### Major Components

#### 🤖 WBR/QBR Automation
- Extract data from multiple sources (Jira, Snowflake, Google Sheets)
- Generate AI-powered insights and trend analysis
- Create PowerPoint presentations automatically
- Schedule and distribute reports

#### 📝 Feature Development Pipeline
- Parse PRD documents to extract requirements
- Generate user stories with AI assistance
- Create Jira epics and stories in bulk
- Maintain traceability from PRD to implementation

#### 📊 Analytics Hub
- ETL pipelines for data warehouse integration
- ML models for PM performance prediction
- Burnout risk detection and prevention
- Interactive dashboards and visualizations

#### 🚨 Real-time Monitoring
- Track PM metrics in real-time
- Anomaly detection and alerting
- Prometheus integration support
- Custom alert rules and thresholds

## 📋 Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment tool (venv, virtualenv, or conda)
- API credentials for the services you want to integrate

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/valdez-nick/obsidian-vault-tools.git
cd obsidian-vault-tools/pm_automation_suite
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install core dependencies
make install

# Or install with all optional features
pip install -e ".[all]"

# Or install specific feature sets
pip install -e ".[ai,integrations,api]"
```

### 4. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
# IMPORTANT: Never commit .env to version control
```

### 5. Configure Credentials

Edit the `.env` file with your actual credentials:

```env
# Essential configurations
OBSIDIAN_VAULT_PATH=/path/to/your/vault
OPENAI_API_KEY=your-openai-key
JIRA_URL=https://your-company.atlassian.net
JIRA_EMAIL=your-email@company.com
JIRA_API_TOKEN=your-jira-token
# ... add other credentials as needed
```

## 🚦 Quick Start

### Running the API Server

```bash
make run-api
# Or directly:
uvicorn pm_automation_suite.api.main:app --reload
```

The API will be available at `http://localhost:8000`

### Running the Scheduler

```bash
make run-scheduler
# Or directly:
python -m pm_automation_suite.orchestration.scheduler
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-coverage

# Run specific test types
pytest tests/unit -v
pytest tests/integration -v
```

## 📖 Usage Examples

### OAuth Authentication

```python
from pm_automation_suite.authentication import AuthenticationManager

# Initialize auth manager
auth_manager = AuthenticationManager()

# Authenticate with Google (browser will open for authorization)
await auth_manager.authenticate(
    provider="google",
    tenant_id="my-workspace",
    client_id="your-client-id",
    client_secret="your-client-secret"
)

# Get valid credential (auto-refreshes if needed)
credential = auth_manager.get_valid_credential("google", "my-workspace")
```

### Basic Connector Usage

```python
from pm_automation_suite.connectors import JiraConnector
from pm_automation_suite.config import Settings

# Load settings
settings = Settings()

# Initialize Jira connector
jira = JiraConnector({
    'jira_url': settings.JIRA_URL,
    'email': settings.JIRA_EMAIL,
    'api_token': settings.JIRA_API_TOKEN
})

# Connect and extract data
jira.connect()
issues = jira.extract_data({
    'jql': 'project = PROJ AND status = "In Progress"',
    'fields': ['summary', 'status', 'assignee']
})
```

### OAuth-Enabled Google Connector

```python
from pm_automation_suite.connectors import GoogleConnectorOAuth

# Initialize with OAuth
google = GoogleConnectorOAuth({
    'tenant_id': 'production',
    'client_id': 'your-google-client-id',
    'client_secret': 'your-google-client-secret'
})

# Connect (will use existing token or prompt for auth)
google.connect()

# Read from Google Sheets
df = google.read_sheet(
    spreadsheet_id='your-sheet-id',
    range_name='Sheet1!A:E'
)
```

### AI Analysis Example

```python
from pm_automation_suite.intelligence import AIAnalyzer
import pandas as pd

# Initialize analyzer
analyzer = AIAnalyzer({
    'openai_api_key': settings.OPENAI_API_KEY,
    'model_preference': 'openai'
})

# Analyze trends
result = await analyzer.analyze_trends(
    data=your_dataframe,
    metrics=['revenue', 'user_count'],
    time_column='date'
)

print(result.insights)
print(result.recommendations)
```

### Workflow Automation Example

```python
from pm_automation_suite.orchestration import WorkflowEngine

# Define workflow
workflow = {
    'name': 'Weekly Report Generation',
    'triggers': ['schedule:weekly'],
    'steps': [
        {'action': 'extract_jira_data', 'params': {...}},
        {'action': 'analyze_metrics', 'params': {...}},
        {'action': 'generate_slides', 'params': {...}},
        {'action': 'send_notification', 'params': {...}}
    ]
}

# Execute workflow
engine = WorkflowEngine()
engine.register_workflow(workflow)
await engine.execute_workflow('Weekly Report Generation')
```

## 🧪 Development

### Setting Up Development Environment

```bash
# Install development dependencies
make install-dev

# Set up pre-commit hooks
make setup-pre-commit

# Run code quality checks
make lint
make format
make type-check
```

### Project Structure

```
pm_automation_suite/
├── authentication/      # OAuth 2.0 authentication manager
├── config/              # Configuration management
├── connectors/          # Data source connectors
├── intelligence/        # AI and analysis modules
├── orchestration/       # Workflow and scheduling
├── templates/           # Document templates
├── tests/               # Test suite
├── api/                 # FastAPI application
└── utils/              # Utility functions
```

### Adding New Connectors

1. Create a new file in `connectors/`
2. Inherit from `BaseConnector`
3. Implement required methods
4. Add configuration to `.env.example`
5. Write tests in `tests/connectors/`

## 🔒 Security

- **OAuth 2.0**: Secure authentication flows with state validation
- **Credential Storage**: Encrypted at rest using Fernet encryption
- **Keyring Integration**: System keyring used for sensitive tokens when available
- **Multi-Tenant Isolation**: Credentials segregated by tenant ID
- **Automatic Token Refresh**: Tokens refreshed automatically before expiry
- Store all credentials in `.env` file (never commit this)
- Use environment-specific configurations
- Implement proper authentication for API endpoints
- Regular security audits with `make security-check`

## 📚 Documentation

### Building Documentation

```bash
# Build docs
make docs

# Serve docs locally
make docs-serve
```

Documentation will be available at `http://localhost:8001`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write comprehensive docstrings
- Add tests for new features
- Run `make check` before committing

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you've activated the virtual environment
2. **API Connection Failures**: Check credentials in `.env`
3. **Missing Dependencies**: Run `pip install -e ".[all]"`

### Debug Mode

Enable debug logging by setting in `.env`:
```env
LOG_LEVEL=DEBUG
DEBUG_MODE=true
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built on top of the Obsidian Vault Tools ecosystem
- Integrates with Atlassian, Google, and Snowflake APIs
- Powered by OpenAI and Anthropic AI models

## 📞 Support

- Create an issue for bug reports
- Check existing issues before creating new ones
- Join discussions for feature requests

---

**Note**: This is an alpha release. APIs and features may change. Use in production with caution.