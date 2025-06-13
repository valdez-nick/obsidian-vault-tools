# Obsidian Librarian v2 - Advanced Features Implementation Tasks

## Overview
Implementation plan for two major features:
1. **Intelligent Tag Management System** - Analyze, clean up, and optimize tag structures
2. **Intelligent Directory Organization System** - Auto-organize notes into appropriate directories

## 🤖 Agent Status (FINAL)
- **tag-models-v2**: ✅ **COMPLETED** - Implemented TagAnalysis, TagSimilarity, TagHierarchy, TagOperation models (commit: 2970c9a)
- **tag-core-v2**: ✅ **COMPLETED** - Built tag_manager.py with all core classes (commit: dbe1116)
- **dir-models-v2**: ✅ **COMPLETED** - Created DirectoryRule, ClassificationResult, OrganizationPlan, MoveOperation models (commit: 5ca45bd)
- **dir-core-v2**: ✅ **COMPLETED** - Implemented auto_organizer.py with ContentClassifier, DirectoryRouter, etc. (commit: 6884388)
- **tag-cli-v2**: ✅ **COMPLETED** - Built comprehensive CLI commands for tag management (commit: f0174ad)

## 📋 Work Trees
- `worktrees/tag-models` → Branch: tag-models
- `worktrees/tag-core-service` → Branch: tag-core-service
- `worktrees/directory-models` → Branch: directory-models
- `worktrees/directory-core-service` → Branch: directory-core-service
- `worktrees/tag-cli-commands` → Branch: tag-cli-commands

---

## Feature 1: Intelligent Tag Management System

### Phase 1: Tag Analysis Engine (Week 1)

#### 🏷️ Core Tag Management Service
- [🤖 CLAIMED - Agent: tag-core] **Create `services/tag_manager.py`**
  - [🤖 CLAIMED] `TagAnalyzer` class for scanning vault tags
  - [🤖 CLAIMED] `TagSimilarityDetector` for finding redundant tags
  - [🤖 CLAIMED] `TagHierarchyBuilder` for suggesting nested structures
  - [🤖 CLAIMED] `AutoTagger` for AI-powered tag suggestions
  - [🤖 CLAIMED] `TagOperations` for bulk rename/merge operations

#### 📊 Tag Analysis Algorithms
- [🤖 CLAIMED - Agent: tag-core] **String Similarity Detection**
  - [🤖 CLAIMED] Implement fuzzy matching for similar tags
  - [🤖 CLAIMED] Case-insensitive comparison (`Python` vs `python`)
  - [🤖 CLAIMED] Pattern matching (`ML`, `machine-learning`, `MachineLearning`)
  
- [🤖 CLAIMED - Agent: tag-core] **Semantic Similarity Analysis**
  - [🤖 CLAIMED] Use embeddings to find conceptually similar tags
  - [🤖 CLAIMED] Detect synonyms and related concepts
  - [🤖 CLAIMED] Build tag relationship graph

- [🤖 CLAIMED - Agent: tag-core] **Usage Pattern Analysis**
  - [🤖 CLAIMED] Track tag co-occurrence patterns
  - [🤖 CLAIMED] Identify frequently used tag combinations
  - [🤖 CLAIMED] Suggest hierarchical relationships

#### 🎯 Content-Based Auto-Tagging
- [🤖 CLAIMED - Agent: tag-core] **Content Analysis Engine**
  - [🤖 CLAIMED] Extract keywords and entities from note content
  - [🤖 CLAIMED] Analyze existing tags to learn tagging patterns
  - [🤖 CLAIMED] Use AI models for topic classification
  - [🤖 CLAIMED] Suggest missing tags for untagged notes

#### 📋 CLI Commands Implementation
- [🤖 CLAIMED - Agent: tag-cli] **Create `cli/tag_commands.py`**
  - [🤖 CLAIMED] `tags analyze` - Comprehensive tag analysis
  - [🤖 CLAIMED] `tags duplicates` - Find similar/redundant tags
  - [🤖 CLAIMED] `tags suggest` - AI-powered tag suggestions
  - [🤖 CLAIMED] `tags auto-tag` - Auto-tag untagged notes
  - [🤖 CLAIMED] `tags merge` - Merge/rename tags safely
  - [🤖 CLAIMED] `tags cleanup` - Interactive cleanup workflow
  - [🤖 CLAIMED] `tags hierarchy` - Suggest tag hierarchies

#### 🔧 Data Models
- [🤖 CLAIMED - Agent: tag-models] **Extend models in `models/models.py`**
  - [🤖 CLAIMED] `TagAnalysis` dataclass
  - [🤖 CLAIMED] `TagSimilarity` dataclass
  - [🤖 CLAIMED] `TagHierarchy` dataclass
  - [🤖 CLAIMED] `TagOperation` dataclass for bulk operations

---

## Feature 2: Intelligent Directory Organization System

### Phase 2: Auto-Organization Engine (Week 2)

#### 📁 Core Organization Service
- [🤖 CLAIMED - Agent: dir-core] **Create `services/auto_organizer.py`**
  - [🤖 CLAIMED] `ContentClassifier` for analyzing note content
  - [🤖 CLAIMED] `DirectoryRouter` for smart file placement
  - [🤖 CLAIMED] `OrganizationLearner` for pattern recognition
  - [🤖 CLAIMED] `RuleEngine` for user-defined rules
  - [🤖 CLAIMED] `FileWatcher` for real-time organization

#### 🧠 Classification Engine
- [🤖 CLAIMED - Agent: dir-core] **Multi-Modal Content Analysis**
  - [🤖 CLAIMED] Text content analysis (topics, keywords, entities)
  - [🤖 CLAIMED] Metadata analysis (tags, frontmatter, titles)
  - [🤖 CLAIMED] Link analysis (connections to other notes)
  - [🤖 CLAIMED] Template pattern detection
  - [🤖 CLAIMED] Date/time pattern recognition

- [🤖 CLAIMED - Agent: dir-core] **Smart Routing Algorithm**
  - [🤖 CLAIMED] Rule-based routing with pattern matching
  - [🤖 CLAIMED] AI-powered content classification
  - [🤖 CLAIMED] Hybrid approach combining rules + AI
  - [🤖 CLAIMED] Conflict resolution for ambiguous cases
  - [🤖 CLAIMED] User feedback learning system

#### 📋 Directory Organization Rules
- [ ] **Configuration System**
  - [ ] Extend `config.yaml` with organization rules
  - [ ] Support for pattern-based routing
  - [ ] Configurable directory structures
  - [ ] Exception handling (Daily Notes ignore rule)
  - [ ] Custom user-defined rules

- [ ] **Default Rule Templates**
  - [ ] Project notes → `Projects/`
  - [ ] Research papers → `Research Library/YYYY/MM/DD/`
  - [ ] Meeting notes → `Meetings/YYYY/MM/`
  - [ ] Knowledge articles → `Knowledge Base/topic/`
  - [ ] Templates → `Templates/` (never move)
  - [ ] Daily notes → `Daily Notes/` (never move)

#### 🎛️ CLI Commands Implementation
- [ ] **Enhance existing `organize` command**
  - [ ] Add `--auto-organize` flag
  - [ ] Add `--learn-patterns` flag
  - [ ] Add `--watch` mode for real-time monitoring
  
- [ ] **Create `cli/organize_commands.py`**
  - [ ] `organize analyze` - Analyze current structure
  - [ ] `organize auto` - Auto-organize misplaced notes
  - [ ] `organize setup` - Interactive rule setup
  - [ ] `organize watch` - Daemon mode for new notes
  - [ ] `organize restructure` - Reorganize entire vault
  - [ ] `organize rules` - Manage organization rules

#### 🔧 Data Models
- [🤖 CLAIMED - Agent: dir-models] **Add models for organization**
  - [🤖 CLAIMED] `DirectoryRule` dataclass
  - [🤖 CLAIMED] `ClassificationResult` dataclass  
  - [🤖 CLAIMED] `OrganizationPlan` dataclass
  - [🤖 CLAIMED] `MoveOperation` dataclass

---

## Phase 3: Integration & Polish (Week 3)

### 🎯 Curate Command Integration
- [ ] **Enhance `curate` command**
  - [ ] Add `--tags` flag for tag cleanup
  - [ ] Add `--organize` flag for directory organization
  - [ ] Integrated workflow: tags → structure → quality
  - [ ] Progress reporting for all operations

### ⚙️ Configuration System
- [ ] **Extend configuration files**
  - [ ] Tag management preferences
  - [ ] Organization rules and patterns
  - [ ] AI model settings for classification
  - [ ] User learning preferences

### 📊 Analytics & Reporting
- [ ] **Organization Metrics**
  - [ ] Track tag cleanup statistics
  - [ ] Monitor organization improvements
  - [ ] Report on vault health scores
  - [ ] Usage pattern analytics

### 🧪 Testing & Quality
- [ ] **Comprehensive Test Suite**
  - [ ] Unit tests for tag analysis algorithms
  - [ ] Integration tests for organization workflows
  - [ ] End-to-end tests with sample vault
  - [ ] Performance tests for large vaults

### 📚 Documentation
- [ ] **User Guides**
  - [ ] Tag management best practices
  - [ ] Directory organization strategies
  - [ ] Configuration examples
  - [ ] Troubleshooting guide

---

## Example CLI Usage

### Tag Management
```bash
# Analyze current tag structure
obsidian-librarian tags analyze /vault/path
# Output: Found 127 tags, 23 potential duplicates, suggest 5 hierarchies

# Find and fix redundant tags
obsidian-librarian tags duplicates /vault/path --merge-similar
# Interactive: "Merge 'Python', 'python', 'PYTHON' → 'python'? (y/n)"

# Auto-tag untagged notes
obsidian-librarian tags auto-tag /vault/path --dry-run
# Output: Would add tags to 45 notes based on content analysis

# Interactive tag cleanup
obsidian-librarian tags cleanup /vault/path --interactive
# Guided workflow through all tag improvements
```

### Directory Organization
```bash
# Analyze and suggest organization improvements
obsidian-librarian organize analyze /vault/path
# Output: 23 misplaced notes, suggest 3 new directories

# Auto-organize misplaced notes
obsidian-librarian organize auto /vault/path --dry-run
# Output: Would move 15 notes to appropriate directories

# Set up organization rules interactively
obsidian-librarian organize setup /vault/path
# Guided setup of patterns and rules

# Watch for new notes and organize automatically
obsidian-librarian organize watch /vault/path --daemon
# Background process monitoring new notes
```

### Integrated Curation
```bash
# Comprehensive vault improvement
obsidian-librarian curate /vault/path --tags --organize --dry-run
# Full workflow: tag cleanup → directory organization → quality improvements

# Interactive comprehensive curation
obsidian-librarian curate /vault/path --tags --organize --interactive
# Step-by-step guided improvements with user approval
```

---

## Success Metrics

### Tag Management
- [ ] Reduce tag redundancy by 80%+
- [ ] Establish consistent tag hierarchy
- [ ] Auto-tag 90%+ of untagged notes accurately
- [ ] Enable bulk tag operations safely

### Directory Organization  
- [ ] Achieve 95%+ accurate auto-classification
- [ ] Reduce misplaced notes to <5%
- [ ] Enable real-time organization
- [ ] Support custom user patterns

### Overall Integration
- [ ] Seamless workflow in `curate` command
- [ ] Comprehensive analytics and reporting
- [ ] Safe operations with backup/rollback
- [ ] User-friendly interactive modes

---

*This task list provides a comprehensive roadmap for implementing advanced tag management and directory organization features in Obsidian Librarian v2.*