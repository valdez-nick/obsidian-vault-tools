#!/usr/bin/env python3
"""
CLI Configuration for Obsidian Vault Manager
Handles command-line arguments and environment variables for AI model configuration
"""

import argparse
import os
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

class CLIConfig:
    """Handles CLI configuration for the vault manager"""
    
    def __init__(self):
        self.user_config_path = Path.home() / ".obsidian_ai_config"
        self.parser = self._create_parser()
        
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser"""
        parser = argparse.ArgumentParser(
            description="AI-Powered Obsidian Vault Manager",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Environment Variables:
  OBSIDIAN_AI_MODEL     Preferred AI model (e.g., dolphin3:latest)
  OLLAMA_HOST          Ollama server URL (default: http://localhost:11434)
  OBSIDIAN_VAULT_PATH  Default vault path

Examples:
  %(prog)s                                    # Auto-detect and run normally
  %(prog)s --model dolphin3:latest            # Use specific model
  %(prog)s --auto-detect                      # Force auto-detection
  %(prog)s --list-models                      # Show available models
  %(prog)s --configure                        # Run interactive configuration
  %(prog)s --vault /path/to/vault             # Use specific vault
  
  # Environment variable usage:
  OBSIDIAN_AI_MODEL=dolphin3:latest %(prog)s
  OLLAMA_HOST=http://192.168.1.100:11434 %(prog)s
            """)
        
        # Model configuration
        parser.add_argument(
            '--model', 
            type=str, 
            help='Specify AI model to use (e.g., dolphin3:latest, llama2:7b)'
        )
        
        parser.add_argument(
            '--auto-detect', 
            action='store_true',
            help='Auto-detect best available model'
        )
        
        parser.add_argument(
            '--list-models', 
            action='store_true',
            help='List available Ollama models and exit'
        )
        
        parser.add_argument(
            '--configure', 
            action='store_true',
            help='Run interactive model configuration'
        )
        
        # Vault configuration
        parser.add_argument(
            '--vault', 
            type=str,
            help='Path to Obsidian vault directory'
        )
        
        # Ollama configuration
        parser.add_argument(
            '--ollama-host', 
            type=str,
            default=os.getenv('OLLAMA_HOST', 'http://localhost:11434'),
            help='Ollama server URL (default: %(default)s)'
        )
        
        # Debug and testing
        parser.add_argument(
            '--debug', 
            action='store_true',
            help='Enable debug logging'
        )
        
        parser.add_argument(
            '--test-models', 
            action='store_true',
            help='Test all available models with sample queries'
        )
        
        parser.add_argument(
            '--reset-config', 
            action='store_true',
            help='Reset saved configuration and run first-time setup'
        )
        
        # Output format
        parser.add_argument(
            '--quiet', '-q', 
            action='store_true',
            help='Minimal output (useful for scripting)'
        )
        
        parser.add_argument(
            '--no-color', 
            action='store_true',
            help='Disable colored output'
        )
        
        return parser
        
    def parse_args(self, args=None):
        """Parse command line arguments"""
        return self.parser.parse_args(args)
        
    def load_user_config(self) -> Dict[str, Any]:
        """Load user configuration from file"""
        if self.user_config_path.exists():
            try:
                with open(self.user_config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load user config: {e}")
        return {}
        
    def save_user_config(self, config: Dict[str, Any]):
        """Save user configuration to file"""
        try:
            with open(self.user_config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Error: Could not save user config: {e}")
            
    def apply_cli_config(self, args) -> Dict[str, Any]:
        """Apply CLI arguments to configuration"""
        config = self.load_user_config()
        
        # Apply command line arguments
        if args.model:
            config['preferred_model'] = args.model
            config['auto_detect'] = False
            
        if args.auto_detect:
            config['auto_detect'] = True
            
        if args.vault:
            config['vault_path'] = args.vault
            
        if args.ollama_host:
            config['ollama_host'] = args.ollama_host
            
        if args.debug:
            config['debug'] = True
            
        if args.quiet:
            config['quiet'] = True
            
        if args.no_color:
            config['no_color'] = True
            
        if args.reset_config:
            # Clear existing config except for CLI overrides
            old_config = config.copy()
            config = {}
            if args.model:
                config['preferred_model'] = old_config.get('preferred_model')
            if args.vault:
                config['vault_path'] = old_config.get('vault_path')
                
        return config
        
    async def handle_special_commands(self, args) -> bool:
        """Handle special CLI commands that don't start the main app"""
        if args.list_models:
            await self._list_models(args.ollama_host)
            return True
            
        if args.test_models:
            await self._test_models(args.ollama_host)
            return True
            
        if args.configure:
            await self._interactive_configure(args.ollama_host)
            return True
            
        return False
        
    async def _list_models(self, ollama_host: str):
        """List available Ollama models"""
        try:
            from models.ollama_adapter import OllamaAdapter
            
            async with OllamaAdapter(ollama_host) as adapter:
                models = adapter.available_models
                
                if models:
                    print(f"\n🤖 Available Models on {ollama_host}:")
                    for i, model in enumerate(models, 1):
                        if "dolphin" in model.lower():
                            print(f"  {i}. 🐬 {model} (Recommended for vault analysis)")
                        elif "llama" in model.lower():
                            print(f"  {i}. 🦙 {model}")
                        elif "mistral" in model.lower():
                            print(f"  {i}. 🌪️  {model}")
                        else:
                            print(f"  {i}. 🤖 {model}")
                    print(f"\nTotal: {len(models)} models")
                else:
                    print(f"❌ No models found on {ollama_host}")
                    print("Install models with: ollama pull <model_name>")
                    
        except Exception as e:
            print(f"❌ Error connecting to Ollama: {e}")
            print(f"Make sure Ollama is running: ollama serve")
            
    async def _test_models(self, ollama_host: str):
        """Test all available models"""
        try:
            from llm_model_manager import LLMModelManager
            
            manager = LLMModelManager()
            models = await manager.auto_detect_models()
            
            if not models:
                print("❌ No models available for testing")
                return
                
            test_query = "What is the capital of France?"
            
            print(f"\n🧪 Testing {len(models)} models with query: '{test_query}'")
            print("=" * 60)
            
            for model in models:
                try:
                    print(f"\n🤖 Testing {model}...")
                    response = await manager.query_model("general_qa", test_query)
                    
                    print(f"✅ Response: {response.content[:100]}...")
                    print(f"⏱️  Time: {response.elapsed_time:.2f}s")
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
                    
        except Exception as e:
            print(f"❌ Testing failed: {e}")
            
    async def _interactive_configure(self, ollama_host: str):
        """Run interactive configuration"""
        try:
            from llm_model_manager import LLMModelManager
            
            manager = LLMModelManager()
            
            print("\n🔧 Interactive AI Model Configuration")
            print("=" * 40)
            
            # Detect models
            models = await manager.auto_detect_models()
            
            if not models:
                print("❌ No models found. Please install a model first:")
                print("  ollama pull dolphin3")
                return
                
            # Let user select model
            selected = await manager.select_model_interactive()
            
            if selected:
                # Additional configuration
                print(f"\n⚙️ Additional Configuration for {selected}")
                
                config = self.load_user_config()
                config.update({
                    'preferred_model': selected,
                    'ollama_host': ollama_host,
                    'auto_detect': False,
                    'configured_at': str(datetime.now())
                })
                
                # Ask about other preferences
                try:
                    default_vault = input("\nDefault vault path (leave empty to ask each time): ").strip()
                    if default_vault:
                        config['vault_path'] = default_vault
                        
                    enable_debug = input("Enable debug logging? (y/N): ").strip().lower() == 'y'
                    config['debug'] = enable_debug
                    
                    self.save_user_config(config)
                    print(f"\n✅ Configuration saved!")
                    print(f"   Model: {selected}")
                    print(f"   Ollama: {ollama_host}")
                    if default_vault:
                        print(f"   Vault: {default_vault}")
                    
                except KeyboardInterrupt:
                    print("\n\nConfiguration cancelled.")
                    
        except Exception as e:
            print(f"❌ Configuration failed: {e}")

# CLI integration function
def main():
    """Main CLI entry point"""
    cli = CLIConfig()
    args = cli.parse_args()
    
    # Handle special commands
    import asyncio
    if asyncio.run(cli.handle_special_commands(args)):
        return
        
    # Apply configuration
    config = cli.apply_cli_config(args)
    
    # Set environment variables for the main app
    if config.get('preferred_model'):
        os.environ['OBSIDIAN_AI_MODEL'] = config['preferred_model']
        
    if config.get('ollama_host'):
        os.environ['OLLAMA_HOST'] = config['ollama_host']
        
    if config.get('vault_path'):
        os.environ['OBSIDIAN_VAULT_PATH'] = config['vault_path']
        
    if config.get('debug'):
        os.environ['DEBUG'] = '1'
        
    # Start the main application
    print("🚀 Starting AI-Powered Obsidian Vault Manager...")
    
    # Import and run the main vault manager
    try:
        from vault_manager_enhanced import EnhancedVaultManager
        
        manager = EnhancedVaultManager()
        manager.run()  # Use the run() method from base class
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    from datetime import datetime
    main()