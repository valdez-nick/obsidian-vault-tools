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

## Technical Debt

See `TECHNICAL_DEBT_REPORT.md` for comprehensive analysis. Key areas:
- 54 files with TODO/FIXME comments
- Duplicate code in `/ai/` and `/models/`
- Generic exception handling needs specificity
- Database performance optimization needed

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

---

*Last updated: After v2.1.0 MCP Interactive Configuration feature*