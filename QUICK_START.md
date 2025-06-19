# 🚀 Quick Start Guide - AI-Powered Obsidian Vault Manager

## Zero-Configuration Setup for Your dolphin3 Model

Since you already have **dolphin3** running on your local machine, the system will automatically detect and configure it for you!

## 1. Quick Start (Recommended)

```bash
# Navigate to the project directory
cd /Users/nvaldez/Documents/repos/Obsidian

# Run with auto-detection (will find your dolphin3 model)
./obsidian-ai

# OR run the enhanced vault manager directly
python vault_manager_enhanced.py
```

The system will:
- ✅ Auto-detect your dolphin3 model
- ✅ Configure it for all AI tasks
- ✅ Show a welcome message on first run
- ✅ Save your preferences for future sessions

## 2. Command Line Options

```bash
# Use a specific model
./obsidian-ai --model dolphin3:latest

# List all available models
./obsidian-ai --list-models

# Interactive configuration
./obsidian-ai --configure

# Test all models
./obsidian-ai --test-models

# Reset configuration and start fresh
./obsidian-ai --reset-config

# Use a specific vault
./obsidian-ai --vault /path/to/your/vault
```

## 3. Environment Variables (Optional)

```bash
# Set preferred model
export OBSIDIAN_AI_MODEL=dolphin3:latest

# Set Ollama host (if not localhost)
export OLLAMA_HOST=http://localhost:11434

# Set default vault path
export OBSIDIAN_VAULT_PATH=/path/to/your/vault

# Run with environment variables
./obsidian-ai
```

## 4. What You'll See on First Run

```
🎉 Welcome to AI-Powered Obsidian Vault Manager!
Setting up your AI assistant...
✅ Found 3 AI models
🐬 Auto-selected: dolphin3:latest (Excellent for vault analysis)
✨ AI Assistant ready with dolphin3:latest
🐬 Dolphin3 is excellent for vault analysis and natural language queries
💡 Try asking: 'What are the main themes in my vault?'
```

## 5. Example AI Queries

Once running, try these natural language queries:

### 📊 Vault Analysis
- "What are the main themes in my vault?"
- "How is my vault organized?"
- "What topics do I write about most?"

### 🔍 Content Search
- "Find all notes about machine learning"
- "Show me recent meeting notes"
- "What projects am I working on?"

### 📝 Summarization
- "Summarize my daily notes from this week"
- "Give me an overview of project Alpha"
- "What are the key points from my research notes?"

### 📈 Statistics
- "How many files are in my vault?"
- "What are my most used tags?"
- "Who is mentioned most in my notes?"

## 6. Configuration Files

The system creates these configuration files automatically:

- `~/.obsidian_ai_config` - Your personal AI preferences
- `model_config.yaml` - AI model configuration (uses dolphin3 by default)
- `llm_feedback/` - Directory for improving AI responses

## 7. Troubleshooting

### If AI models aren't detected:
```bash
# Check if Ollama is running
ollama list

# Start Ollama if needed
ollama serve

# Check if dolphin3 is available
ollama pull dolphin3
```

### If you want to reconfigure:
```bash
# Reset and reconfigure
./obsidian-ai --reset-config --configure
```

### For debugging:
```bash
# Run with debug output
./obsidian-ai --debug
```

## 8. Features You Get with dolphin3

🐬 **dolphin3** is automatically configured for:
- **Intent Classification** - Understanding what you're asking
- **Question Answering** - Natural conversation about your vault
- **Code Analysis** - Understanding technical content
- **Summarization** - Creating concise summaries
- **Entity Extraction** - Finding people, dates, projects

## 9. Privacy & Security

✅ **100% Local Processing** - Your data never leaves your machine  
✅ **No Internet Required** - Works completely offline  
✅ **Encrypted Vault Compatible** - Works with encrypted Obsidian vaults  
✅ **Open Source Models** - Full transparency in AI processing  

## 10. Advanced Usage

### Rating Responses
```
Query: What are my main research topics?
[AI provides response]
> rate 5 excellent analysis
✓ Thank you! Your rating helps improve AI responses.
```

### Model Management
Access through the main menu: **Option 8 - AI Model Configuration**
- View model performance stats
- Pull new models
- Configure preferences
- Export training data

## Need Help?

- 📖 Full documentation: `LLM_SETUP_GUIDE.md`
- 🐛 Issues: Check the troubleshooting section
- 💡 Features: See `LLM_INTEGRATION_SUMMARY.md`

**Ready to explore your vault with AI? Run `./obsidian-ai` and start asking questions!** 🚀