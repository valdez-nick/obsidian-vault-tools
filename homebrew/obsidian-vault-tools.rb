class ObsidianVaultTools < Formula
  include Language::Python::Virtualenv

  desc "Comprehensive toolkit for managing Obsidian vaults"
  homepage "https://github.com/yourusername/obsidian-vault-tools"
  url "https://github.com/yourusername/obsidian-vault-tools/archive/refs/tags/v1.0.0.tar.gz"
  sha256 "YOUR_SHA256_HERE"
  license "MIT"
  head "https://github.com/yourusername/obsidian-vault-tools.git", branch: "main"

  depends_on "python@3.11"
  depends_on "rust" => :build  # For potential Rust extensions
  depends_on "pygame"  # For audio support
  depends_on "pillow"  # For image processing

  resource "click" do
    url "https://files.pythonhosted.org/packages/source/c/click/click-8.1.3.tar.gz"
    sha256 "7682dc8afb30297001674575ea00d1814d808d6a36af415a82bd481d37ba7b8e"
  end

  resource "rich" do
    url "https://files.pythonhosted.org/packages/source/r/rich/rich-13.3.5.tar.gz"
    sha256 "2d11b9b8dd03868f09b4fffadc84a6a8cda574e40dc90821bd845720ebb8e89c"
  end

  resource "pyyaml" do
    url "https://files.pythonhosted.org/packages/source/P/PyYAML/PyYAML-6.0.tar.gz"
    sha256 "68fb519c14306fec9720a2a5b45bc9f0c8d1b9c72adf45c37baedfcd949c35a2"
  end

  resource "aiohttp" do
    url "https://files.pythonhosted.org/packages/source/a/aiohttp/aiohttp-3.8.4.tar.gz"
    sha256 "bf2e1a9162c1e441bf805a1fd166e249d574ca04e03b34f97e2928769e91ab5c"
  end

  def install
    virtualenv_install_with_resources

    # Install shell completions
    generate_completions_from_executable(bin/"obsidian-tools", shells: [:bash, :zsh, :fish])
  end

  test do
    # Test basic functionality
    assert_match "Obsidian Vault Tools", shell_output("#{bin}/obsidian-tools --help")
    assert_match version.to_s, shell_output("#{bin}/obsidian-tools version")
    
    # Test CLI alias
    assert_match "Obsidian Vault Tools", shell_output("#{bin}/ovt --help")
  end

  def caveats
    <<~EOS
      Obsidian Vault Tools has been installed!

      Quick start:
        ovt config set-vault ~/Documents/YourVault
        ovt --help

      For AI features, you'll need to install additional dependencies:
        pip install transformers sentence-transformers torch

      Audio features require pygame to be properly configured.
      If you experience audio issues on macOS, try:
        brew reinstall pygame
    EOS
  end
end