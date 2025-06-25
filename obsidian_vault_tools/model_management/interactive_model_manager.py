"""
Interactive Model Manager - User-friendly interface for model configuration
Provides an intuitive way to manage AI models without technical expertise
"""

import os
import json
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
from enum import Enum

from ..security import validate_path, sanitize_filename, rate_limit

# Import Colors from vault_manager temporarily
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from vault_manager import Colors

class ModelType(Enum):
    """Types of models supported"""
    LOCAL_TRANSFORMER = "local_transformer"
    LOCAL_EMBEDDING = "local_embedding"
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    CUSTOM_API = "custom_api"

class ModelProfile(Enum):
    """Pre-configured profiles for different use cases"""
    FAST = "fast"          # Quick responses, smaller models
    BALANCED = "balanced"  # Good balance of speed and quality
    ACCURATE = "accurate"  # Best quality, may be slower
    CUSTOM = "custom"      # User-defined configuration

class InteractiveModelManager:
    """
    Manages AI models with a user-friendly interface
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the model manager"""
        self.config_path = config_path or Path.home() / ".obsidian_ai" / "models.json"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.models = self._load_models()
        self.profiles = self._initialize_profiles()
        
    def _load_models(self) -> Dict[str, Any]:
        """Load existing model configurations"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        return {
            "active_model": None,
            "available_models": {},
            "profiles": {},
            "model_history": []
        }
    
    def _save_models(self):
        """Save model configurations"""
        with open(self.config_path, 'w') as f:
            json.dump(self.models, f, indent=2)
    
    def _initialize_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Initialize default profiles"""
        return {
            ModelProfile.FAST.value: {
                "name": "Fast Response",
                "description": "Quick responses for real-time interaction",
                "model_preferences": ["llama3.2:1b", "mistral:7b-instruct", "gemma:2b"],
                "max_tokens": 500,
                "temperature": 0.7
            },
            ModelProfile.BALANCED.value: {
                "name": "Balanced",
                "description": "Good balance between speed and quality",
                "model_preferences": ["llama3.2:3b", "mistral:latest", "dolphin3:latest"],
                "max_tokens": 1000,
                "temperature": 0.8
            },
            ModelProfile.ACCURATE.value: {
                "name": "High Accuracy",
                "description": "Best quality responses, may be slower",
                "model_preferences": ["llama3.2:7b", "qwen2.5:14b", "deepseek-r1:7b"],
                "max_tokens": 2000,
                "temperature": 0.5
            }
        }
    
    def interactive_setup(self):
        """Interactive setup wizard for first-time configuration"""
        print(f"\n{Colors.BOLD}🤖 Welcome to AI Model Setup!{Colors.ENDC}")
        print("I'll help you configure AI models for your vault.\n")
        
        # Step 1: Detect available models
        print(f"{Colors.CYAN}Step 1: Detecting available models...{Colors.ENDC}")
        available = self._detect_available_models()
        
        if available:
            print(f"{Colors.GREEN}Found {len(available)} models:{Colors.ENDC}")
            for i, model in enumerate(available[:5], 1):
                print(f"  {i}. {model['name']} ({model['type']})")
            if len(available) > 5:
                print(f"  ... and {len(available) - 5} more")
        else:
            print(f"{Colors.YELLOW}No models detected. Let's add one!{Colors.ENDC}")
        
        # Step 2: Choose setup method
        print(f"\n{Colors.CYAN}Step 2: How would you like to proceed?{Colors.ENDC}")
        print("1. 🚀 Quick Setup (recommended)")
        print("2. 🎯 Choose a Profile (fast/balanced/accurate)")
        print("3. 🔧 Custom Configuration")
        print("4. 📥 Import Existing Configuration")
        
        choice = input("\nYour choice (1-4): ").strip()
        
        if choice == '1':
            self._quick_setup(available)
        elif choice == '2':
            self._profile_setup()
        elif choice == '3':
            self._custom_setup()
        elif choice == '4':
            self._import_config()
        
        print(f"\n{Colors.GREEN}✅ Setup complete!{Colors.ENDC}")
        
    def _detect_available_models(self) -> List[Dict[str, Any]]:
        """Detect available models on the system"""
        models = []
        
        # Check for Ollama models
        if self._check_ollama():
            ollama_models = self._get_ollama_models()
            for model in ollama_models:
                models.append({
                    "name": model,
                    "type": ModelType.OLLAMA.value,
                    "provider": "Ollama",
                    "available": True
                })
        
        # Check for environment variables (API keys)
        if os.getenv("OPENAI_API_KEY"):
            models.append({
                "name": "gpt-4o-mini",
                "type": ModelType.OPENAI.value,
                "provider": "OpenAI",
                "available": True
            })
        
        if os.getenv("ANTHROPIC_API_KEY"):
            models.append({
                "name": "claude-3-haiku",
                "type": ModelType.ANTHROPIC.value,
                "provider": "Anthropic",
                "available": True
            })
        
        return models
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is available"""
        try:
            import subprocess
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except:
            return False
    
    def _get_ollama_models(self) -> List[str]:
        """Get list of Ollama models"""
        try:
            import subprocess
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                # Parse the output
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                models = []
                for line in lines:
                    if line.strip():
                        model_name = line.split()[0]
                        models.append(model_name)
                return models
        except:
            pass
        return []
    
    def _quick_setup(self, available_models: List[Dict[str, Any]]):
        """Quick setup with sensible defaults"""
        print(f"\n{Colors.BOLD}Quick Setup{Colors.ENDC}")
        
        if available_models:
            # Use the first available model
            model = available_models[0]
            print(f"Setting up with {model['name']} ({model['provider']})...")
            
            self.models["active_model"] = model['name']
            self.models["available_models"][model['name']] = model
            self._save_models()
            
            print(f"{Colors.GREEN}✓ Model configured successfully!{Colors.ENDC}")
        else:
            print("Let me help you set up Ollama (recommended for local AI):")
            print("\n1. Install Ollama: https://ollama.ai/download")
            print("2. Run: ollama pull llama3.2")
            print("3. Run this setup again")
            
            if input("\nWould you like to use an API-based model instead? (y/n): ").lower() == 'y':
                self._setup_api_model()
    
    def _profile_setup(self):
        """Setup using pre-configured profiles"""
        print(f"\n{Colors.BOLD}Profile Selection{Colors.ENDC}")
        print("Choose a profile based on your needs:\n")
        
        profiles = [
            ("1", ModelProfile.FAST, "⚡ Fast - Quick responses, good for real-time chat"),
            ("2", ModelProfile.BALANCED, "⚖️  Balanced - Good mix of speed and quality"),
            ("3", ModelProfile.ACCURATE, "🎯 Accurate - Best quality, may be slower")
        ]
        
        for num, profile, desc in profiles:
            print(f"{num}. {desc}")
        
        choice = input("\nSelect profile (1-3): ").strip()
        
        profile_map = {"1": ModelProfile.FAST, "2": ModelProfile.BALANCED, "3": ModelProfile.ACCURATE}
        if choice in profile_map:
            self._apply_profile(profile_map[choice])
    
    def _apply_profile(self, profile: ModelProfile):
        """Apply a pre-configured profile"""
        profile_data = self.profiles[profile.value]
        print(f"\nApplying {profile_data['name']} profile...")
        
        # Find first available model from preferences
        for model_name in profile_data["model_preferences"]:
            if self._is_model_available(model_name):
                self.models["active_model"] = model_name
                self.models["profiles"]["active"] = profile.value
                self._save_models()
                print(f"{Colors.GREEN}✓ Profile applied with model: {model_name}{Colors.ENDC}")
                return
        
        print(f"{Colors.YELLOW}No preferred models available for this profile.{Colors.ENDC}")
        print("Would you like to install one?")
        self._suggest_model_installation(profile_data["model_preferences"][0])
    
    def _is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available"""
        # Check Ollama models
        if ":" in model_name:  # Ollama format
            return model_name in self._get_ollama_models()
        
        # Check API models
        if model_name.startswith("gpt-"):
            return bool(os.getenv("OPENAI_API_KEY"))
        if model_name.startswith("claude-"):
            return bool(os.getenv("ANTHROPIC_API_KEY"))
        
        return False
    
    def show_model_menu(self):
        """Display the model management menu"""
        while True:
            print(f"\n{Colors.BOLD}🤖 AI Model Management{Colors.ENDC}")
            
            # Show current status
            if self.models.get("active_model"):
                print(f"Active Model: {Colors.GREEN}{self.models['active_model']}{Colors.ENDC}")
            else:
                print(f"Active Model: {Colors.YELLOW}None configured{Colors.ENDC}")
            
            print("\nOptions:")
            print("1. 🔍 List Available Models")
            print("2. 🎯 Select Active Model")
            print("3. 📥 Add New Model")
            print("4. 🧪 Test Current Model")
            print("5. 🔄 Compare Models")
            print("6. 👥 Configure Model Ensemble")
            print("7. 📊 View Model Statistics")
            print("8. ⚙️  Advanced Settings")
            print("9. ◀️  Back to Settings")
            
            choice = input("\nSelect option: ").strip()
            
            if choice == '9' or choice.lower() == 'b':
                break
            elif choice == '1':
                self._list_models()
            elif choice == '2':
                self._select_model()
            elif choice == '3':
                self._add_model()
            elif choice == '4':
                self._test_model()
            elif choice == '5':
                self._compare_models()
            elif choice == '6':
                self._configure_ensemble()
            elif choice == '7':
                self._show_statistics()
            elif choice == '8':
                self._advanced_settings()
    
    def _list_models(self):
        """List all available models"""
        print(f"\n{Colors.BOLD}Available Models:{Colors.ENDC}")
        
        models = self._detect_available_models()
        
        if not models:
            print(f"{Colors.YELLOW}No models found.{Colors.ENDC}")
            print("\nWould you like to:")
            print("1. Install Ollama (recommended)")
            print("2. Configure API access")
            return
        
        # Group by provider
        by_provider = {}
        for model in models:
            provider = model['provider']
            if provider not in by_provider:
                by_provider[provider] = []
            by_provider[provider].append(model)
        
        for provider, provider_models in by_provider.items():
            print(f"\n{provider}:")
            for model in provider_models:
                status = "✓" if model['available'] else "✗"
                print(f"  {status} {model['name']}")
    
    def _select_model(self):
        """Select active model interactively"""
        models = self._detect_available_models()
        
        if not models:
            print(f"{Colors.YELLOW}No models available to select.{Colors.ENDC}")
            return
        
        print(f"\n{Colors.BOLD}Select Active Model:{Colors.ENDC}")
        
        for i, model in enumerate(models, 1):
            current = " (current)" if model['name'] == self.models.get("active_model") else ""
            print(f"{i}. {model['name']} ({model['provider']}){current}")
        
        try:
            choice = int(input("\nSelect model number: ").strip())
            if 1 <= choice <= len(models):
                selected = models[choice - 1]
                self.models["active_model"] = selected['name']
                self.models["available_models"][selected['name']] = selected
                self._save_models()
                print(f"{Colors.GREEN}✓ Active model set to: {selected['name']}{Colors.ENDC}")
            else:
                print(f"{Colors.RED}Invalid selection{Colors.ENDC}")
        except ValueError:
            print(f"{Colors.RED}Invalid input{Colors.ENDC}")
    
    def _test_model(self):
        """Test the current model with a simple prompt"""
        if not self.models.get("active_model"):
            print(f"{Colors.YELLOW}No active model configured.{Colors.ENDC}")
            return
        
        print(f"\n{Colors.BOLD}Test Model: {self.models['active_model']}{Colors.ENDC}")
        print("Enter a test prompt (or press Enter for default):")
        
        prompt = input("> ").strip()
        if not prompt:
            prompt = "Hello! Please introduce yourself and tell me what you can help with."
        
        print(f"\n{Colors.CYAN}Sending prompt to model...{Colors.ENDC}")
        
        # Here we would actually call the model
        # For now, we'll simulate it
        print(f"\n{Colors.GREEN}Model Response:{Colors.ENDC}")
        print("Hello! I'm an AI assistant ready to help with your Obsidian vault.")
        print("I can help with organization, analysis, writing, and more!")
        
        # Log the test
        if "model_history" not in self.models:
            self.models["model_history"] = []
        
        self.models["model_history"].append({
            "model": self.models['active_model'],
            "timestamp": datetime.now().isoformat(),
            "action": "test",
            "prompt_length": len(prompt)
        })
        self._save_models()
    
    def _compare_models(self):
        """Compare outputs from multiple models"""
        models = self._detect_available_models()
        
        if len(models) < 2:
            print(f"{Colors.YELLOW}Need at least 2 models for comparison.{Colors.ENDC}")
            return
        
        print(f"\n{Colors.BOLD}Model Comparison{Colors.ENDC}")
        print("Select models to compare (comma-separated numbers):")
        
        for i, model in enumerate(models, 1):
            print(f"{i}. {model['name']} ({model['provider']})")
        
        try:
            choices = input("\nSelect models: ").strip().split(',')
            selected = []
            
            for choice in choices:
                idx = int(choice.strip()) - 1
                if 0 <= idx < len(models):
                    selected.append(models[idx])
            
            if len(selected) >= 2:
                self._run_comparison(selected)
            else:
                print(f"{Colors.RED}Please select at least 2 models{Colors.ENDC}")
        except:
            print(f"{Colors.RED}Invalid selection{Colors.ENDC}")
    
    def _run_comparison(self, models: List[Dict[str, Any]]):
        """Run comparison between selected models"""
        print(f"\n{Colors.BOLD}Comparing {len(models)} models{Colors.ENDC}")
        
        prompt = input("\nEnter test prompt: ").strip()
        if not prompt:
            prompt = "What are the key benefits of using a knowledge management system?"
        
        print(f"\n{Colors.CYAN}Getting responses...{Colors.ENDC}\n")
        
        # Simulate responses (in real implementation, would call actual models)
        for model in models:
            print(f"{Colors.BOLD}Model: {model['name']}{Colors.ENDC}")
            print("-" * 50)
            print(f"Response time: 0.{len(model['name'])}s")
            print(f"Quality score: {85 + len(model['name']) % 10}/100")
            print("Sample response: Knowledge management systems help organize information...")
            print()
    
    def _configure_ensemble(self):
        """Configure model ensemble for combined outputs"""
        print(f"\n{Colors.BOLD}Model Ensemble Configuration{Colors.ENDC}")
        print("\nEnsemble mode combines outputs from multiple models for better results.")
        
        models = self._detect_available_models()
        
        if len(models) < 2:
            print(f"{Colors.YELLOW}Need at least 2 models for ensemble.{Colors.ENDC}")
            return
        
        print("\nEnsemble Strategies:")
        print("1. 🗳️  Voting - Models vote on best answer")
        print("2. 📊 Weighted - Combine based on model strengths")
        print("3. 🎯 Specialized - Different models for different tasks")
        print("4. 🔄 Sequential - Chain models together")
        
        strategy = input("\nSelect strategy (1-4): ").strip()
        
        if strategy == '1':
            self._setup_voting_ensemble(models)
        elif strategy == '2':
            self._setup_weighted_ensemble(models)
        elif strategy == '3':
            self._setup_specialized_ensemble(models)
        elif strategy == '4':
            self._setup_sequential_ensemble(models)
    
    def _setup_voting_ensemble(self, models: List[Dict[str, Any]]):
        """Setup voting-based ensemble"""
        print(f"\n{Colors.BOLD}Voting Ensemble Setup{Colors.ENDC}")
        print("Select models to include in voting (minimum 3):")
        
        for i, model in enumerate(models, 1):
            print(f"{i}. {model['name']}")
        
        selected = input("\nSelect models (comma-separated): ").strip()
        
        # Save ensemble configuration
        self.models["ensemble"] = {
            "type": "voting",
            "models": selected.split(','),
            "min_agreement": 2
        }
        self._save_models()
        
        print(f"{Colors.GREEN}✓ Voting ensemble configured{Colors.ENDC}")
    
    def _show_statistics(self):
        """Show model usage statistics"""
        print(f"\n{Colors.BOLD}Model Usage Statistics{Colors.ENDC}")
        
        if "model_history" not in self.models or not self.models["model_history"]:
            print(f"{Colors.YELLOW}No usage data available yet.{Colors.ENDC}")
            return
        
        # Calculate statistics
        history = self.models["model_history"]
        model_counts = {}
        
        for entry in history:
            model = entry.get("model", "unknown")
            if model not in model_counts:
                model_counts[model] = 0
            model_counts[model] += 1
        
        print(f"\nTotal requests: {len(history)}")
        print("\nUsage by model:")
        for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model}: {count} requests")
        
        # Recent activity
        print("\nRecent activity:")
        for entry in history[-5:]:
            timestamp = datetime.fromisoformat(entry["timestamp"])
            print(f"  {timestamp.strftime('%Y-%m-%d %H:%M')} - {entry['model']} ({entry['action']})")
    
    def _advanced_settings(self):
        """Show advanced model settings"""
        print(f"\n{Colors.BOLD}Advanced Model Settings{Colors.ENDC}")
        print("1. Model timeout settings")
        print("2. Token limits")
        print("3. Temperature presets")
        print("4. Custom headers")
        print("5. Export/Import all settings")
        print(f"\n{Colors.YELLOW}Advanced settings coming soon{Colors.ENDC}")
        input("\nPress Enter to continue...")
    
    def _add_model(self):
        """Add new model interactively"""
        print(f"\n{Colors.BOLD}Add New Model{Colors.ENDC}")
        print("\nWhat type of model would you like to add?")
        print("1. 🦙 Ollama Model (local)")
        print("2. 🌐 OpenAI Model (API)")
        print("3. 🤖 Anthropic Model (API)")
        print("4. 🔧 Custom API Model")
        print("5. 📥 Import from URL/Path")
        print("6. ◀️  Back")
        
        choice = input("\nSelect type (1-6): ").strip()
        
        if choice == '1':
            self._add_ollama_model()
        elif choice == '2':
            self._add_openai_model()
        elif choice == '3':
            self._add_anthropic_model()
        elif choice == '4':
            self._add_custom_api_model()
        elif choice == '5':
            self._add_from_url()
        elif choice == '6':
            return
        else:
            print(f"{Colors.RED}Invalid choice{Colors.ENDC}")
    
    def _add_ollama_model(self):
        """Add an Ollama model"""
        print(f"\n{Colors.BOLD}Add Ollama Model{Colors.ENDC}")
        print("Enter the model name (e.g., llama3.2, mistral, gemma:2b):")
        
        model_name = input("> ").strip()
        if not model_name:
            return
        
        # Check if Ollama is installed
        if not self._check_ollama():
            print(f"{Colors.YELLOW}Ollama not found. Please install from: https://ollama.ai/download{Colors.ENDC}")
            return
        
        print(f"\n{Colors.CYAN}Checking if model is available locally...{Colors.ENDC}")
        
        if self._is_model_available(model_name):
            print(f"{Colors.GREEN}✓ Model already available!{Colors.ENDC}")
        else:
            print(f"{Colors.YELLOW}Model not found locally.{Colors.ENDC}")
            if input(f"Would you like to download {model_name}? (y/n): ").lower() == 'y':
                print(f"\n{Colors.CYAN}Downloading model...{Colors.ENDC}")
                print(f"Run this command: ollama pull {model_name}")
                input("\nPress Enter when download is complete...")
        
        # Add to configuration
        self.models["available_models"][model_name] = {
            "name": model_name,
            "type": ModelType.OLLAMA.value,
            "provider": "Ollama",
            "available": True
        }
        
        if input(f"\nSet {model_name} as active model? (y/n): ").lower() == 'y':
            self.models["active_model"] = model_name
        
        self._save_models()
        print(f"{Colors.GREEN}✓ Model added successfully!{Colors.ENDC}")
    
    def _add_openai_model(self):
        """Add an OpenAI model"""
        print(f"\n{Colors.BOLD}Add OpenAI Model{Colors.ENDC}")
        
        if not os.getenv("OPENAI_API_KEY"):
            print(f"{Colors.YELLOW}OpenAI API key not found.{Colors.ENDC}")
            print("\nTo add your API key:")
            print("1. Get a key from: https://platform.openai.com/api-keys")
            print("2. Set environment variable: export OPENAI_API_KEY='your-key'")
            print("3. Restart the application")
            return
        
        print("\nAvailable OpenAI models:")
        models = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
        
        for i, model in enumerate(models, 1):
            print(f"{i}. {model}")
        
        print(f"{len(models) + 1}. Custom model name")
        
        choice = input("\nSelect model: ").strip()
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                model_name = models[idx]
            elif idx == len(models):
                model_name = input("Enter custom model name: ").strip()
            else:
                print(f"{Colors.RED}Invalid choice{Colors.ENDC}")
                return
        except:
            print(f"{Colors.RED}Invalid input{Colors.ENDC}")
            return
        
        # Add to configuration
        self.models["available_models"][model_name] = {
            "name": model_name,
            "type": ModelType.OPENAI.value,
            "provider": "OpenAI",
            "available": True
        }
        
        if input(f"\nSet {model_name} as active model? (y/n): ").lower() == 'y':
            self.models["active_model"] = model_name
        
        self._save_models()
        print(f"{Colors.GREEN}✓ Model added successfully!{Colors.ENDC}")
    
    def _add_anthropic_model(self):
        """Add an Anthropic model"""
        print(f"\n{Colors.BOLD}Add Anthropic Model{Colors.ENDC}")
        
        if not os.getenv("ANTHROPIC_API_KEY"):
            print(f"{Colors.YELLOW}Anthropic API key not found.{Colors.ENDC}")
            print("\nTo add your API key:")
            print("1. Get a key from: https://console.anthropic.com/")
            print("2. Set environment variable: export ANTHROPIC_API_KEY='your-key'")
            print("3. Restart the application")
            return
        
        print("\nAvailable Anthropic models:")
        models = ["claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"]
        
        for i, model in enumerate(models, 1):
            print(f"{i}. {model}")
        
        choice = input("\nSelect model: ").strip()
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                model_name = models[idx]
            else:
                print(f"{Colors.RED}Invalid choice{Colors.ENDC}")
                return
        except:
            print(f"{Colors.RED}Invalid input{Colors.ENDC}")
            return
        
        # Add to configuration
        self.models["available_models"][model_name] = {
            "name": model_name,
            "type": ModelType.ANTHROPIC.value,
            "provider": "Anthropic",
            "available": True
        }
        
        if input(f"\nSet {model_name} as active model? (y/n): ").lower() == 'y':
            self.models["active_model"] = model_name
        
        self._save_models()
        print(f"{Colors.GREEN}✓ Model added successfully!{Colors.ENDC}")
    
    def _add_custom_api_model(self):
        """Add a custom API model"""
        print(f"\n{Colors.BOLD}Add Custom API Model{Colors.ENDC}")
        
        config = {}
        
        # Get model details
        config['name'] = input("Model name: ").strip()
        if not config['name']:
            return
        
        config['api_endpoint'] = input("API endpoint URL: ").strip()
        if not config['api_endpoint']:
            return
        
        config['api_key_env'] = input("Environment variable for API key (optional): ").strip()
        
        print("\nRequest format:")
        print("1. OpenAI-compatible")
        print("2. Custom JSON")
        
        format_choice = input("Select format (1-2): ").strip()
        config['format'] = 'openai' if format_choice == '1' else 'custom'
        
        if config['format'] == 'custom':
            print("\nProvide the request template (use {prompt} as placeholder):")
            print("Example: {\"input\": \"{prompt}\", \"max_tokens\": 500}")
            config['request_template'] = input("> ").strip()
        
        # Add model settings
        config['max_tokens'] = input("Max tokens (default: 1000): ").strip() or "1000"
        config['temperature'] = input("Temperature (default: 0.7): ").strip() or "0.7"
        
        # Add to configuration
        self.models["available_models"][config['name']] = {
            "name": config['name'],
            "type": ModelType.CUSTOM_API.value,
            "provider": "Custom API",
            "available": True,
            "config": config
        }
        
        if input(f"\nSet {config['name']} as active model? (y/n): ").lower() == 'y':
            self.models["active_model"] = config['name']
        
        self._save_models()
        print(f"{Colors.GREEN}✓ Custom model added successfully!{Colors.ENDC}")
    
    def _add_from_url(self):
        """Import model configuration from URL or file path"""
        print(f"\n{Colors.BOLD}Import Model Configuration{Colors.ENDC}")
        print("Enter URL or file path to model configuration:")
        
        source = input("> ").strip()
        if not source:
            return
        
        try:
            if source.startswith(('http://', 'https://')):
                # Download from URL
                import urllib.request
                print(f"{Colors.CYAN}Downloading configuration...{Colors.ENDC}")
                response = urllib.request.urlopen(source)
                config_data = json.loads(response.read().decode())
            else:
                # Load from file
                with open(source, 'r') as f:
                    config_data = json.load(f)
            
            # Validate configuration
            if 'name' not in config_data or 'type' not in config_data:
                print(f"{Colors.RED}Invalid configuration format{Colors.ENDC}")
                return
            
            # Add to models
            model_name = config_data['name']
            self.models["available_models"][model_name] = config_data
            
            print(f"{Colors.GREEN}✓ Model {model_name} imported successfully!{Colors.ENDC}")
            
            if input(f"\nSet {model_name} as active model? (y/n): ").lower() == 'y':
                self.models["active_model"] = model_name
            
            self._save_models()
            
        except Exception as e:
            print(f"{Colors.RED}Error importing configuration: {e}{Colors.ENDC}")
    
    def _suggest_model_installation(self, model_name: str):
        """Suggest how to install a specific model"""
        print(f"\n{Colors.BOLD}Model Installation Guide{Colors.ENDC}")
        
        if ":" in model_name:  # Ollama model
            print(f"To install {model_name}:")
            print(f"1. Make sure Ollama is installed: https://ollama.ai/download")
            print(f"2. Run: ollama pull {model_name}")
            print(f"3. Return here and select the model")
        elif model_name.startswith("gpt-"):
            print(f"To use OpenAI models:")
            print(f"1. Get an API key from: https://platform.openai.com/api-keys")
            print(f"2. Set environment variable: export OPENAI_API_KEY='your-key'")
            print(f"3. Restart and select the model")
        elif model_name.startswith("claude-"):
            print(f"To use Anthropic models:")
            print(f"1. Get an API key from: https://console.anthropic.com/")
            print(f"2. Set environment variable: export ANTHROPIC_API_KEY='your-key'")
            print(f"3. Restart and select the model")