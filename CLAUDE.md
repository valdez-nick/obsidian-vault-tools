# CLAUDE.md - Development Guide

## Project Overview

Obsidian Vault Tools is a comprehensive toolkit for managing Obsidian vaults with AI features, MCP integration, and a unified interactive interface. The project has evolved from a collection of separate tools into a cohesive toolsuite with a single entry point.

## Recent Major Updates

### v2.0.0 - Unified Toolsuite Release (Completed)
- Transformed from separate tools to unified interface
- Single entry point (`ovt`) for all features
- 11 organized menu categories
- Dynamic feature detection
- Comprehensive security overhaul
- MCP tools integrated into menu system

### v2.1.0 - MCP Interactive Configuration (In PR #2)
- Added interactive MCP server configuration to Settings menu
- No more manual JSON editing required
- Guided wizard for adding servers
- Secure credential management with masking
- Connection testing and validation
- Single example template for demonstration

### v2.2.0 - PM Tools Integration (2025-06-24)
- Integrated PM burnout prevention tools into unified manager
- Added Burnout Detection Analysis to PM Tools menu
- Added Content Quality Analysis to Analysis menu (placeholder for future ContentQualityEngine)
- Added Generate Enhanced Daily Note to AI & LLM Features menu
- Implemented basic PM daily note generation with WSJF priorities
- Full support for TaskExtractor, WSJFAnalyzer, EisenhowerMatrixClassifier, and BurnoutDetector
- Graceful fallback for planned features (ContentQualityEngine, daily_template_generator)
- Integrated with existing burnout detection system
- Completes the PM burnout prevention toolkit

### v2.2.1 - Menu Navigation Fix (2025-06-26)
- Fixed critical MenuNavigator error that prevented the interactive menu from loading
- Issue: `navigate_menu()` was missing required positional argument 'options'
- Root cause: Method was being called with incorrect parameters format
- Solution: Properly format menu options as tuples of (key, description) before passing to MenuNavigator
- Feature status improved from 11/17 to 13/17 enabled features
- Missing features identified: analysis, backup, v2, and content_quality modules

### v2.2.2 - Path Resolution & Configuration Fixes (2025-06-26)
- **Fixed vault path resolution bug**: Paths like "Users/nvaldez/Documents/Obsidian Vault" no longer get appended to current working directory
- **Added persistent vault path configuration**: Tool now remembers vault paths between sessions
- **Enhanced MCP setup experience**: Added Docker requirement checks and detailed Atlassian setup guide
- **New configuration commands**: Added `ovt config set-vault`, `ovt config show`, and `ovt config reset`
- **MCP requirements checker**: Added `ovt mcp check-requirements` to verify system dependencies
- **Smart path resolution**: Automatically detects and fixes common path input issues

## Architecture

### Core Components

#### 1. **Unified Manager** (`unified_vault_manager.py`)
- Central interactive interface
- Dynamic feature loading with graceful fallbacks
- Menu-driven navigation with arrow key support
- Integrates all tools and features

Key classes:
- `UnifiedVaultManager`: Main application class
- Feature detection system
- Menu handlers for each category

#### 2. **MCP Integration** (`obsidian_vault_tools/mcp_tools/`)
- Dynamic tool discovery and execution
- Multi-server support
- Interactive configuration (NEW in v2.1.0)
- Secure credential management

Key modules:
- `client_manager.py`: Server lifecycle management
- `config.py`: Configuration storage and validation
- `interactive_config.py`: Interactive configuration UI (NEW)
- `tools/discovery.py`: Tool discovery service
- `tools/executor.py`: Tool execution engine
- `tools/menu_builder.py`: Dynamic menu generation

#### 3. **Security** (`obsidian_vault_tools/security.py`)
- Comprehensive security utilities
- Input validation and sanitization
- Path traversal prevention
- Rate limiting implementation
- Credential encryption

#### 4. **Feature Modules**
- `ai/`: AI and LLM integrations
- `analysis/`: Vault analysis tools
- `audio/`: Sound effects and ambiance
- `backup/`: Backup management
- `creative/`: ASCII art and flowcharts
- `organization/`: Tag and file organization
- `pm_tools/`: PM-specific burnout prevention tools (NEW)

## Development Guidelines

### Adding New Features

1. **Create Feature Module**
   ```python
   # In appropriate directory
   class NewFeature:
       def __init__(self, vault_path):
           self.vault_path = vault_path
   ```

2. **Add to Unified Manager**
   ```python
   # In unified_vault_manager.py
   try:
       from new_module import NewFeature
       NEW_FEATURE_AVAILABLE = True
   except ImportError:
       NEW_FEATURE_AVAILABLE = False
   ```

3. **Integrate into Menu**
   - Add to appropriate menu category
   - Create handler method
   - Update feature detection

4. **Follow Patterns**
   - Use Colors class for output
   - Handle missing dependencies gracefully
   - Add to feature status display

### Security Considerations

1. **Always Validate Input**
   ```python
   from obsidian_vault_tools.security import validate_path, sanitize_filename
   
   path = validate_path(user_input, base_path=vault_path)
   filename = sanitize_filename(user_filename)
   ```

2. **Mask Sensitive Data**
   - Never display tokens, passwords in plain text
   - Use credential manager for storage
   - Implement masking in UI displays

3. **Safe Command Execution**
   - Never use `shell=True` without validation
   - Use `shlex.split()` for command parsing
   - Validate commands before execution

### Testing Guidelines

1. **Test Missing Dependencies**
   ```bash
   # Test with minimal install
   pip install -e .
   ovt  # Should work with reduced features
   ```

2. **Test Menu Integration**
   - Navigate all menu paths
   - Test back/cancel operations
   - Verify error handling

3. **Security Testing**
   - Test path traversal attempts
   - Verify credential masking
   - Check input validation

## Feature Roadmap

### High Priority
- [ ] Complete subprocess security fixes
- [ ] Add database connection pooling
- [ ] Consolidate duplicate code directories
- [ ] Implement missing AI features

### Medium Priority
- [ ] Enhanced error recovery
- [ ] Performance optimizations
- [ ] Extended MCP server templates
- [ ] Plugin system architecture

### Future Enhancements
- [ ] Web interface option
- [ ] Mobile companion app
- [ ] Cloud sync capabilities
- [ ] Collaborative features

## Recent Technical Implementations

### v2.2.2 Technical Details

#### Path Resolution Fix
**Problem**: `Path.resolve()` in `security.py` was resolving relative paths against current working directory.
**Solution**: Added smart path detection in `unified_vault_manager.py:332-334`
```python
if vault_path and not vault_path.startswith(('/', '~', '.')) and vault_path.startswith('Users/'):
    vault_path = '/' + vault_path
```

#### Configuration Persistence
**Implementation**: Enhanced `_get_vault_path()` method with Config integration
- Priority order: Environment Variable → Saved Config → User Prompt
- Automatic saving of validated paths: `unified_vault_manager.py:359-362`
- Config file location: `~/.obsidian_vault_tools.json`

#### CLI Command Extensions
**Added commands**: `obsidian_vault_tools/cli.py:464-529`
- `ovt config set-vault <path>` - Set default vault path
- `ovt config show` - Display current configuration
- `ovt config reset` - Reset to defaults
- `ovt mcp check-requirements` - Verify system dependencies

#### MCP Setup Improvements
**Enhanced Atlassian setup**: `mcp_tools/setup_wizard.py:152-214`
- Docker availability checking using `shutil.which('docker')`
- Detailed setup instructions with URLs for API token creation
- Credential storage with masking
- Connection readiness verification

#### Security Enhancements
- Maintained existing security validation in `validate_path()`
- Added input sanitization before path processing
- Preserved credential masking in MCP configuration

## Technical Debt

See `TECHNICAL_DEBT_REPORT.md` for comprehensive analysis. Key areas:
- 54 files with TODO/FIXME comments
- Duplicate code in `/ai/` and `/models/`
- Generic exception handling needs specificity
- Database performance optimization needed
- Missing feature modules (as of v2.2.1):
  - `analysis/` module components not fully available
  - `backup/` module components missing
  - `obsidian_librarian_v2` module not found
  - `ContentQualityEngine` from pm_tools not yet implemented

## Contributing

### Workflow
1. Create feature branch from `main`
2. Follow existing code patterns
3. Add comprehensive documentation
4. Include tests where applicable
5. Create detailed PR with examples

### Code Standards
- Use type hints where possible
- Follow PEP 8 guidelines
- Document public methods
- Handle exceptions gracefully
- Add security validation

### PR Requirements
- Clear description of changes
- Examples of usage
- Documentation updates
- No breaking changes (or clearly marked)
- Security implications noted

## Maintenance Notes

### Documentation Updates
- Update `README.md` for user-facing changes
- Keep `UNIFIED_MANAGER_README.md` current
- Update `MCP_CONFIG_GUIDE.md` for MCP changes
- Add to `CLAUDE.md` for architectural changes

### Release Process
1. Merge feature branches to main
2. Update version in `setup.py`
3. Create annotated tag
4. Generate release notes
5. Publish to PyPI

### Version Numbering
- Major: Breaking changes
- Minor: New features
- Patch: Bug fixes

## Development Environment

### Recommended Setup
```bash
# Clone repository
git clone https://github.com/valdez-nick/obsidian-vault-tools.git
cd obsidian-vault-tools

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in development mode
pip install -e ".[dev]"

# Install optional dependencies
pip install -e ".[ai,mcp,audio]"
```

### Environment Variables
```bash
export OBSIDIAN_VAULT_PATH="/path/to/vault"
export OPENAI_API_KEY="your-key"  # For AI features
export DEBUG=true  # Enable debug logging
```

## Support

- GitHub Issues: Bug reports and feature requests
- Discussions: General questions and ideas
- Wiki: Extended documentation and guides

## PM Tools Usage Examples

### Content Quality Engine
The Content Quality Engine helps maintain consistency across PM notes:

```python
from obsidian_vault_tools.pm_tools import ContentQualityEngine

# Initialize and run quality checks
quality_engine = ContentQualityEngine(vault_path)
report = quality_engine.analyze_vault()

# Get specific quality metrics
quality_score = report['overall_score']
naming_issues = report['naming_inconsistencies']
incomplete_notes = report['incomplete_content']
```

Key features:
- Detects naming inconsistencies (e.g., "DFP" vs "Device Fingerprinting")
- Identifies incomplete thoughts and sentences
- Finds duplicate content across files
- Provides standardization suggestions
- Generates quality scores for prioritization

### Daily Template Generator
The Daily Template Generator creates PM-optimized daily notes:

```python
from obsidian_vault_tools.pm_tools import DailyTemplateGenerator

# Generate today's template
generator = DailyTemplateGenerator(vault_path)
template = generator.generate_template()

# Generate with custom date
template = generator.generate_template(date="2025-06-25")
```

Template includes:
- Top 3 WSJF priorities for the day
- Product area focus rotation
- Energy and context tracking sections
- Completion rate monitoring
- Meeting notes structure
- End-of-day reflection prompts

Both tools integrate seamlessly with the existing PM burnout prevention suite, working alongside the WSJF prioritizer, Eisenhower matrix, and burnout detection system.

## v2.3.0 - PM Automation Suite Integration

### Overview
The PM Automation Suite is a comprehensive enterprise-grade automation platform for Product Management workflows. It provides end-to-end automation from data extraction to insight generation, feature development, and real-time monitoring.

### Architecture

#### Core Infrastructure
- **OAuth 2.0 Authentication**: Multi-provider support (Google, Microsoft, Atlassian)
- **Secure Credential Management**: Keyring integration with encryption at rest
- **Event-Driven Orchestration**: Pub/sub event bus for workflow coordination
- **Resilient Connectors**: Rate limiting, retry logic, and connection pooling

#### Major Components

1. **WBR/QBR Automation**
   ```python
   from wbr import WBRWorkflow
   
   workflow = WBRWorkflow(config)
   await workflow.run_complete_workflow()
   ```
   - Extracts data from Jira, Snowflake, Google Sheets
   - Generates AI-powered insights with statistical analysis
   - Creates PowerPoint presentations automatically
   - Schedules and distributes reports

2. **Feature Development Pipeline**
   ```python
   from feature_pipeline import FeaturePipeline
   
   pipeline = FeaturePipeline(config)
   await pipeline.process_prd("path/to/prd.pdf")
   ```
   - Parses PRD documents (PDF, Word, Markdown)
   - Extracts requirements and acceptance criteria
   - Generates user stories with AI assistance
   - Bulk creates Jira issues with proper linking

3. **Analytics Hub**
   ```python
   from analytics_hub import ETLPipeline, PMPerformancePredictor
   
   # Run ETL
   pipeline = ETLPipeline(config)
   await pipeline.execute_pipeline()
   
   # ML predictions
   predictor = PMPerformancePredictor({'metric': 'velocity'})
   prediction = predictor.predict(data)
   ```
   - ETL pipelines with multiple data sources
   - ML models for performance prediction
   - Burnout risk detection algorithms
   - Interactive dashboard generation

4. **Real-time Monitoring**
   ```python
   from analytics_hub import MonitoringSystem
   
   monitoring = MonitoringSystem(config)
   monitoring.create_default_pm_monitoring()
   await monitoring.start_monitoring()
   ```
   - Real-time metric collection
   - Anomaly detection (statistical, threshold, trend)
   - Alert management with severity levels
   - Prometheus integration support

### UI Integration

The PM Automation Suite is fully integrated into the Unified Vault Manager:

```
PM Tools & Task Management
├── Extract Tasks from Vault
├── WSJF Priority Analysis
├── Eisenhower Matrix Classification
├── Burnout Detection Analysis
├── Combined PM Dashboard
├── Export PM Reports
└── ──── PM Automation Suite ────
    ├── 🤖 WBR/QBR Automation
    ├── 📝 Feature Development Pipeline
    ├── 📊 Analytics Hub & ML Insights
    ├── 🚨 Real-time Monitoring
    └── ⚙️ PM Suite Configuration
```

### Testing & Quality

- **Test Coverage**: 158 tests with 80+ passing
- **Unit Tests**: Core functionality validation
- **Integration Tests**: API and service integration
- **Performance Tests**: Response time and scalability
- **Security Tests**: Input validation and credential handling

### Configuration

Configuration is managed through environment variables and secure credential storage:

```bash
# Jira Configuration
JIRA_URL=https://your-domain.atlassian.net
JIRA_EMAIL=your-email@company.com

# Snowflake Configuration
SNOWFLAKE_ACCOUNT=your-account
SNOWFLAKE_WAREHOUSE=your-warehouse

# AI Providers
OPENAI_API_KEY=your-key
ANTHROPIC_API_KEY=your-key
```

### Dependencies

Core dependencies:
- `aiohttp`: Async HTTP client
- `pandas`: Data manipulation
- `scikit-learn`: ML models
- `plotly`: Interactive visualizations
- `python-pptx`: PowerPoint generation
- `tenacity`: Retry logic
- `pydantic`: Data validation

Optional dependencies:
- `tensorflow`: Deep learning models
- `prometheus-client`: Metrics export
- `snowflake-connector-python`: Snowflake integration

### Future Enhancements

1. **Enhanced AI Integration**
   - Custom fine-tuned models for PM tasks
   - Multi-modal analysis (images, diagrams)
   - Real-time collaboration features

2. **Extended Integrations**
   - Slack/Teams notifications
   - GitHub/GitLab integration
   - Additional data warehouses

3. **Advanced Analytics**
   - Predictive sprint planning
   - Resource optimization algorithms
   - Team performance analytics

### Migration Guide

For users upgrading from v2.2.x:

1. Install new dependencies:
   ```bash
   pip install -e ".[pm-automation]"
   ```

2. Configure credentials:
   ```bash
   ovt
   # Navigate to: Settings → PM Suite Configuration
   ```

3. Test connections:
   ```bash
   # In the PM Suite menu
   # Select: Test All Connections
   ```

The PM Automation Suite represents a major advancement in PM tooling, providing enterprise-ready automation while maintaining the simplicity and elegance of the Obsidian Vault Tools ecosystem.

---

*Last updated: After v2.3.0 PM Automation Suite Integration*