# Ubuntu 24.04 LTS Installation Guide for Obsidian Vault Tools

This guide provides comprehensive installation instructions specifically for Ubuntu 24.04 LTS users, including all system dependencies, troubleshooting, and best practices.

## Table of Contents
- [System Requirements](#system-requirements)
- [Quick Install](#quick-install)
- [Detailed Installation](#detailed-installation)
- [Virtual Environment Setup](#virtual-environment-setup)
- [Feature-Specific Dependencies](#feature-specific-dependencies)
- [Post-Installation Setup](#post-installation-setup)
- [Troubleshooting](#troubleshooting)
- [Verification Steps](#verification-steps)

## System Requirements

Ubuntu 24.04 LTS ships with Python 3.12, which exceeds the minimum requirement of Python 3.8+.

### Hardware Requirements
- **Minimum RAM**: 4GB (8GB recommended for AI features)
- **Disk Space**: 2GB for full installation with all features
- **GPU**: Optional - NVIDIA GPU with CUDA support for accelerated AI features

## Quick Install

For users who want to get started quickly with all features:

```bash
# Update system and install all dependencies
sudo apt update && sudo apt upgrade -y

# Install all system dependencies at once
sudo apt install -y \
    python3-pip python3-venv python3-dev build-essential \
    libssl-dev libffi-dev \
    libsdl2-dev libsdl2-mixer-dev libsdl2-image-dev libsdl2-ttf-dev \
    libportaudio2 portaudio19-dev python3-pygame \
    libopenblas-dev liblapack-dev gfortran \
    libjpeg-dev zlib1g-dev libpng-dev \
    git curl wget \
    docker.io docker-compose \
    redis-server libpq-dev

# Install Node.js 20 LTS (required for MCP)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# Configure Docker permissions
sudo usermod -aG docker $USER
newgrp docker

# Install Ollama for local LLM support (optional)
curl -fsSL https://ollama.com/install.sh | sh

# Create virtual environment and install obsidian-vault-tools
python3 -m venv ~/obsidian-tools-env
source ~/obsidian-tools-env/bin/activate
pip install --upgrade pip setuptools wheel
pip install obsidian-vault-tools[all]
```

## Detailed Installation

### Step 1: System Package Dependencies

#### Core Development Tools
```bash
# Essential build tools and Python development headers
sudo apt update
sudo apt install -y python3-pip python3-venv python3-dev build-essential

# SSL and cryptography libraries (required for secure connections)
sudo apt install -y libssl-dev libffi-dev

# Version control
sudo apt install -y git
```

#### Audio Support (SDL2 and PortAudio)
Required for the dungeon crawler-themed sound effects:

```bash
# SDL2 libraries for pygame
sudo apt install -y \
    libsdl2-dev \
    libsdl2-mixer-dev \
    libsdl2-image-dev \
    libsdl2-ttf-dev

# PortAudio for cross-platform audio
sudo apt install -y libportaudio2 portaudio19-dev

# Python pygame package (system-wide)
sudo apt install -y python3-pygame

# Additional audio support
sudo apt install -y pulseaudio pavucontrol
```

#### Node.js and npm (for MCP features)
```bash
# Install Node.js 20 LTS (recommended for MCP compatibility)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# Verify installation
node --version  # Should show v20.x.x
npm --version   # Should show 10.x.x
```

#### Docker (for Atlassian Integration)
```bash
# Install Docker and Docker Compose
sudo apt install -y docker.io docker-compose

# Start and enable Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add current user to docker group (IMPORTANT for permissions)
sudo usermod -aG docker $USER

# Apply group changes (or logout/login)
newgrp docker
```

#### ML/AI Libraries
```bash
# BLAS and LAPACK for numerical computations
sudo apt install -y libopenblas-dev liblapack-dev gfortran

# Image processing libraries (for Pillow/PIL)
sudo apt install -y \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libtiff-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libwebp-dev

# Additional ML dependencies
sudo apt install -y libhdf5-dev
```

### Step 2: Optional System Dependencies

```bash
# Redis for caching
sudo apt install -y redis-server
sudo systemctl enable redis-server

# PostgreSQL client libraries
sudo apt install -y libpq-dev

# Additional utilities
sudo apt install -y \
    curl \
    wget \
    jq \
    htop
```

## Virtual Environment Setup

Using a virtual environment is **strongly recommended** on Ubuntu to avoid conflicts with system Python packages:

```bash
# Create a dedicated directory for the project
mkdir -p ~/projects/obsidian-tools
cd ~/projects/obsidian-tools

# Create virtual environment using Python 3.12
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip, setuptools, and wheel in the virtual environment
pip install --upgrade pip setuptools wheel

# Install obsidian-vault-tools with all features
pip install obsidian-vault-tools[all]
```

### Creating an Activation Script

For convenience, create a script to activate your environment:

```bash
# Create activation script
cat > ~/activate-obsidian-tools.sh << 'EOF'
#!/bin/bash
source ~/projects/obsidian-tools/venv/bin/activate
echo "Obsidian Vault Tools environment activated!"
echo "Run 'ovt' to start the unified manager"
EOF

chmod +x ~/activate-obsidian-tools.sh

# Add alias to .bashrc for easy activation
echo "alias ovt-env='source ~/activate-obsidian-tools.sh'" >> ~/.bashrc
source ~/.bashrc
```

## Feature-Specific Dependencies

### Local LLM Support (Ollama)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model (e.g., llama2)
ollama pull llama2

# Start Ollama service
sudo systemctl start ollama
sudo systemctl enable ollama
```

### GPU Support (NVIDIA CUDA)

For accelerated AI features with NVIDIA GPUs:

```bash
# Check if you have an NVIDIA GPU
lspci | grep -i nvidia

# If you have an NVIDIA GPU, install drivers and CUDA
sudo apt install -y nvidia-driver-535 nvidia-cuda-toolkit

# Reboot after driver installation
sudo reboot

# After reboot, verify CUDA installation
nvidia-smi
nvcc --version
```

## Post-Installation Setup

### 1. Docker Permissions

After adding your user to the docker group, you must either:
- Log out and log back in, OR
- Run `newgrp docker` in your current terminal

Verify Docker works without sudo:
```bash
docker run hello-world
```

### 2. Configure Obsidian Vault Tools

```bash
# Set default vault path
ovt config set-vault ~/Documents/ObsidianVault

# Check MCP requirements
ovt mcp check-requirements

# Configure MCP servers interactively
ovt
# Navigate to: Settings & Configuration → MCP Server Configuration
```

### 3. Environment Variables

Create a `.env` file for API keys and configurations:

```bash
# Create .env file in your project directory
cat > ~/projects/obsidian-tools/.env << 'EOF'
# OpenAI API (for AI features)
OPENAI_API_KEY=your-openai-api-key

# Anthropic API (optional)
ANTHROPIC_API_KEY=your-anthropic-api-key

# GitHub Token (for MCP GitHub integration)
GITHUB_PERSONAL_ACCESS_TOKEN=your-github-token

# Atlassian (for Confluence/Jira integration)
JIRA_URL=https://your-domain.atlassian.net
JIRA_EMAIL=your-email@company.com
JIRA_API_TOKEN=your-atlassian-api-token

# Vault Configuration
OBSIDIAN_VAULT_PATH=~/Documents/ObsidianVault
EOF

# Set appropriate permissions
chmod 600 ~/projects/obsidian-tools/.env
```

### 4. Audio Configuration

If audio isn't working:

```bash
# Ensure PulseAudio is running
pulseaudio --check || pulseaudio --start

# Test audio
speaker-test -t wav -c 2

# If needed, install additional audio packages
sudo apt install -y alsa-utils
```

## Troubleshooting

### Common Issues and Solutions

#### 1. "pygame.error: No available audio device"

```bash
# Solution: Install and configure PulseAudio
sudo apt install -y pulseaudio pavucontrol
pulseaudio --kill
pulseaudio --start

# Alternative: Set SDL audio driver
export SDL_AUDIODRIVER=pulse
```

#### 2. "Permission denied" when running Docker commands

```bash
# Solution: Ensure user is in docker group
groups $USER  # Check if 'docker' is listed

# If not, add user to docker group and re-login
sudo usermod -aG docker $USER
# Then logout and login again, or run:
su - $USER
```

#### 3. "ModuleNotFoundError: No module named 'mcp'"

```bash
# Solution: Install with MCP support
pip install obsidian-vault-tools[mcp]
# Or install all features:
pip install obsidian-vault-tools[all]
```

#### 4. Node.js version issues with MCP

```bash
# Solution: Ensure Node.js 18+ is installed
node --version

# If version is too old, reinstall:
sudo apt remove nodejs
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

#### 5. SSL Certificate errors

```bash
# Solution: Update certificates
sudo apt install -y ca-certificates
sudo update-ca-certificates
```

#### 6. Virtual environment not activating

```bash
# Solution: Use full path
source ~/projects/obsidian-tools/venv/bin/activate

# Or check if venv was created correctly
python3 -m venv --help
```

### Performance Issues

#### Slow AI operations
- Ensure you have sufficient RAM (8GB+ recommended)
- Consider using Ollama with smaller models
- Check if GPU acceleration is enabled (for NVIDIA GPUs)

#### High CPU usage
```bash
# Monitor system resources
htop

# Limit parallel operations in settings
ovt config set max-workers 2
```

## Verification Steps

After installation, verify everything is working:

```bash
# 1. Check Python version
python3 --version  # Should be 3.12.x on Ubuntu 24.04

# 2. Verify obsidian-vault-tools installation
pip show obsidian-vault-tools

# 3. Test the CLI
ovt --version

# 4. Check all features are available
ovt check-features

# 5. Test Docker (if using MCP Atlassian)
docker run hello-world

# 6. Test Node.js (if using MCP)
npm --version
node --version

# 7. Test audio system
python3 -c "import pygame; pygame.mixer.init(); print('Audio OK')"

# 8. Launch the unified manager
ovt

# 9. Run MCP requirements check
ovt mcp check-requirements
```

### Expected Output

If everything is installed correctly, `ovt check-features` should show:
- ✅ Core features enabled
- ✅ AI features available (if installed)
- ✅ MCP integration ready (if installed)
- ✅ PM Automation Suite available (if installed)
- ✅ Audio system functional

## Updating

To update obsidian-vault-tools:

```bash
# Activate virtual environment
source ~/projects/obsidian-tools/venv/bin/activate

# Update the package
pip install --upgrade obsidian-vault-tools[all]

# Update system dependencies if needed
sudo apt update
sudo apt upgrade
```

## Uninstallation

To completely remove obsidian-vault-tools:

```bash
# Remove Python package
pip uninstall obsidian-vault-tools

# Remove virtual environment
rm -rf ~/projects/obsidian-tools/venv

# Remove configuration
rm -rf ~/.obsidian_vault_tools
rm -f ~/.obsidian_vault_tools.json

# Optional: Remove system packages (be careful, other apps might need these)
# sudo apt remove --purge docker.io nodejs
```

## Getting Help

If you encounter issues not covered in this guide:

1. Check the [main documentation](README.md)
2. Review [GitHub Issues](https://github.com/valdez-nick/obsidian-vault-tools/issues)
3. Enable debug mode: `export DEBUG=true`
4. Run with verbose output: `ovt --verbose`

## Next Steps

1. [Configure MCP servers](SETUP.md) for extended functionality
2. [Set up PM Automation](pm_automation_suite/docs/QUICK_START.md) for PM features
3. Explore the [unified manager](UNIFIED_MANAGER_README.md) interface
4. Read about [AI integration](LLM_SETUP_GUIDE.md) options

---

*This guide is specifically tailored for Ubuntu 24.04 LTS. For other distributions or versions, some commands may need adjustment.*