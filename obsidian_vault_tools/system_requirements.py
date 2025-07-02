#!/usr/bin/env python3
"""
System requirements checker for Obsidian Vault Tools.
Detects missing system dependencies and provides platform-specific installation instructions.
"""

import platform
import subprocess
import shutil
import os
import sys
from typing import Dict, List, Tuple, Optional
from pathlib import Path

class SystemRequirementsChecker:
    """Check and validate system requirements for Obsidian Vault Tools."""
    
    def __init__(self):
        self.platform = platform.system().lower()
        self.distro = self._get_linux_distribution()
        self.missing_deps = []
        self.warnings = []
        
    def _get_linux_distribution(self) -> Optional[str]:
        """Get the Linux distribution name."""
        if self.platform != 'linux':
            return None
            
        try:
            # Try to read from /etc/os-release (standard on modern Linux)
            if Path('/etc/os-release').exists():
                with open('/etc/os-release', 'r') as f:
                    for line in f:
                        if line.startswith('ID='):
                            return line.split('=')[1].strip().strip('"')
        except:
            pass
            
        # Fallback to lsb_release
        if shutil.which('lsb_release'):
            try:
                result = subprocess.run(['lsb_release', '-i'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    return result.stdout.split(':')[1].strip().lower()
            except:
                pass
                
        return 'unknown'
    
    def check_python_version(self) -> Tuple[bool, str]:
        """Check if Python version meets requirements."""
        version = sys.version_info
        min_version = (3, 8)
        
        if version >= min_version:
            return True, f"Python {version.major}.{version.minor}.{version.micro}"
        else:
            return False, f"Python {version.major}.{version.minor} (3.8+ required)"
    
    def check_command_available(self, command: str) -> bool:
        """Check if a command is available in PATH."""
        return shutil.which(command) is not None
    
    def check_package_installed(self, package: str) -> bool:
        """Check if a system package is installed (Ubuntu/Debian specific)."""
        if self.distro not in ['ubuntu', 'debian']:
            return True  # Assume installed on non-Ubuntu systems
            
        try:
            result = subprocess.run(['dpkg', '-s', package], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def check_audio_support(self) -> Dict[str, bool]:
        """Check audio system dependencies."""
        audio_deps = {
            'libsdl2-dev': False,
            'libsdl2-mixer-dev': False,
            'libportaudio2': False,
            'portaudio19-dev': False,
            'pulseaudio': False
        }
        
        if self.platform == 'linux' and self.distro in ['ubuntu', 'debian']:
            for dep in audio_deps:
                audio_deps[dep] = self.check_package_installed(dep)
                
        # Check if pygame can initialize audio
        try:
            import pygame
            pygame.mixer.init()
            pygame.mixer.quit()
            audio_deps['pygame_functional'] = True
        except:
            audio_deps['pygame_functional'] = False
            
        return audio_deps
    
    def check_docker(self) -> Dict[str, any]:
        """Check Docker installation and permissions."""
        docker_info = {
            'installed': self.check_command_available('docker'),
            'running': False,
            'user_in_group': False,
            'version': None
        }
        
        if not docker_info['installed']:
            return docker_info
            
        # Check if Docker daemon is running
        try:
            result = subprocess.run(['docker', 'ps'], 
                                  capture_output=True, text=True)
            docker_info['running'] = result.returncode == 0
            
            # If permission denied, user likely not in docker group
            if 'permission denied' in result.stderr.lower():
                docker_info['user_in_group'] = False
            else:
                docker_info['user_in_group'] = True
                
        except:
            pass
            
        # Get Docker version
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                docker_info['version'] = result.stdout.strip()
        except:
            pass
            
        return docker_info
    
    def check_nodejs(self) -> Dict[str, any]:
        """Check Node.js and npm installation."""
        node_info = {
            'node_installed': self.check_command_available('node'),
            'npm_installed': self.check_command_available('npm'),
            'node_version': None,
            'npm_version': None,
            'version_ok': False
        }
        
        if node_info['node_installed']:
            try:
                result = subprocess.run(['node', '--version'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    version = result.stdout.strip()
                    node_info['node_version'] = version
                    # Check if version is 14+
                    major_version = int(version.split('.')[0].lstrip('v'))
                    node_info['version_ok'] = major_version >= 14
            except:
                pass
                
        if node_info['npm_installed']:
            try:
                result = subprocess.run(['npm', '--version'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    node_info['npm_version'] = result.stdout.strip()
            except:
                pass
                
        return node_info
    
    def check_virtual_environment(self) -> Dict[str, any]:
        """Check if running in a virtual environment."""
        venv_info = {
            'in_venv': False,
            'venv_path': None,
            'recommended': True
        }
        
        # Check various virtual environment indicators
        if hasattr(sys, 'real_prefix'):
            venv_info['in_venv'] = True
            venv_info['venv_path'] = sys.prefix
        elif hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix:
            venv_info['in_venv'] = True
            venv_info['venv_path'] = sys.prefix
        elif os.environ.get('VIRTUAL_ENV'):
            venv_info['in_venv'] = True
            venv_info['venv_path'] = os.environ.get('VIRTUAL_ENV')
            
        return venv_info
    
    def get_ubuntu_install_commands(self) -> List[str]:
        """Get Ubuntu-specific installation commands for missing dependencies."""
        commands = []
        
        # System update
        commands.append("sudo apt update")
        
        # Python development tools
        if not self.check_package_installed('python3-dev'):
            commands.append("sudo apt install -y python3-dev build-essential")
        
        # Audio dependencies
        audio_deps = self.check_audio_support()
        missing_audio = [pkg for pkg, installed in audio_deps.items() 
                        if not installed and pkg != 'pygame_functional']
        if missing_audio:
            commands.append(f"sudo apt install -y {' '.join(missing_audio)}")
        
        # Docker
        docker_info = self.check_docker()
        if not docker_info['installed']:
            commands.append("sudo apt install -y docker.io docker-compose")
            commands.append("sudo systemctl enable docker")
            commands.append("sudo systemctl start docker")
        
        if docker_info['installed'] and not docker_info['user_in_group']:
            commands.append(f"sudo usermod -aG docker $USER")
            commands.append("# NOTE: Log out and back in for Docker group to take effect")
        
        # Node.js
        node_info = self.check_nodejs()
        if not node_info['node_installed'] or not node_info['version_ok']:
            commands.extend([
                "curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -",
                "sudo apt install -y nodejs"
            ])
        
        return commands
    
    def check_all(self) -> Dict[str, any]:
        """Run all system checks."""
        results = {
            'platform': self.platform,
            'distro': self.distro,
            'python': self.check_python_version(),
            'virtual_env': self.check_virtual_environment(),
            'audio': self.check_audio_support(),
            'docker': self.check_docker(),
            'nodejs': self.check_nodejs(),
            'missing_deps': [],
            'warnings': []
        }
        
        # Compile missing dependencies
        if not all(results['audio'].values()):
            results['missing_deps'].append('Audio libraries (SDL2, PortAudio)')
            
        if not results['docker']['installed']:
            results['missing_deps'].append('Docker')
        elif not results['docker']['user_in_group']:
            results['warnings'].append('User not in docker group')
            
        if not results['nodejs']['node_installed']:
            results['missing_deps'].append('Node.js')
        elif not results['nodejs']['version_ok']:
            results['warnings'].append('Node.js version too old (14+ required)')
            
        if not results['virtual_env']['in_venv']:
            results['warnings'].append('Not running in virtual environment (recommended)')
        
        return results
    
    def print_report(self):
        """Print a formatted system requirements report."""
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        
        console = Console()
        
        results = self.check_all()
        
        # Create status table
        table = Table(title="System Requirements Check")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details")
        
        # Python version
        py_ok, py_version = results['python']
        table.add_row(
            "Python", 
            "✓" if py_ok else "✗",
            py_version
        )
        
        # Virtual environment
        venv = results['virtual_env']
        table.add_row(
            "Virtual Environment",
            "✓" if venv['in_venv'] else "⚠",
            venv['venv_path'] if venv['in_venv'] else "Not active (recommended)"
        )
        
        # Audio support
        audio = results['audio']
        audio_ok = audio.get('pygame_functional', False)
        table.add_row(
            "Audio Support",
            "✓" if audio_ok else "✗",
            "Pygame audio functional" if audio_ok else "Missing audio libraries"
        )
        
        # Docker
        docker = results['docker']
        docker_status = "✓" if docker['installed'] and docker['user_in_group'] else "⚠" if docker['installed'] else "✗"
        docker_details = docker['version'] if docker['installed'] else "Not installed"
        if docker['installed'] and not docker['user_in_group']:
            docker_details += " (user not in docker group)"
        table.add_row("Docker", docker_status, docker_details)
        
        # Node.js
        node = results['nodejs']
        node_status = "✓" if node['node_installed'] and node['version_ok'] else "⚠" if node['node_installed'] else "✗"
        node_details = f"Node {node['node_version']}, npm {node['npm_version']}" if node['node_installed'] else "Not installed"
        table.add_row("Node.js", node_status, node_details)
        
        console.print(table)
        
        # Show missing dependencies
        if results['missing_deps']:
            console.print("\n[red]Missing Dependencies:[/red]")
            for dep in results['missing_deps']:
                console.print(f"  • {dep}")
        
        # Show warnings
        if results['warnings']:
            console.print("\n[yellow]Warnings:[/yellow]")
            for warning in results['warnings']:
                console.print(f"  • {warning}")
        
        # Show installation commands for Ubuntu
        if self.distro in ['ubuntu', 'debian'] and (results['missing_deps'] or results['warnings']):
            commands = self.get_ubuntu_install_commands()
            if commands:
                console.print("\n[green]Ubuntu Installation Commands:[/green]")
                console.print(Panel("\n".join(commands), title="Run these commands"))
        
        return results


def check_requirements():
    """Main entry point for system requirements checking."""
    checker = SystemRequirementsChecker()
    checker.print_report()


if __name__ == "__main__":
    check_requirements()