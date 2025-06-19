#!/usr/bin/env python3
"""
File Versioning System
Prevents overwriting of existing documents by creating versioned files
"""

import os
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

class FileVersioning:
    """
    Handles file versioning to prevent overwriting existing documents
    """
    
    @staticmethod
    def get_next_version_path(file_path: str, prefix: str = "", suffix: str = "") -> str:
        """
        Generate a versioned file path that doesn't conflict with existing files
        
        Args:
            file_path: Original file path
            prefix: Optional prefix to add to filename
            suffix: Optional suffix to add before extension
            
        Returns:
            New file path with version number
        """
        path = Path(file_path)
        directory = path.parent
        stem = path.stem
        extension = path.suffix
        
        # Add prefix and suffix if provided
        if prefix:
            stem = f"{prefix}_{stem}"
        if suffix:
            stem = f"{stem}_{suffix}"
        
        # Check if file exists
        new_path = directory / f"{stem}{extension}"
        
        if not new_path.exists():
            return str(new_path)
        
        # File exists, need to version it
        version = 1
        
        # Look for existing version pattern
        existing_files = list(directory.glob(f"{stem}_v*{extension}"))
        if existing_files:
            # Extract version numbers
            version_numbers = []
            for existing_file in existing_files:
                match = re.search(r'_v(\d+)', existing_file.stem)
                if match:
                    version_numbers.append(int(match.group(1)))
            
            if version_numbers:
                version = max(version_numbers) + 1
        
        # Generate versioned filename
        versioned_name = f"{stem}_v{version:02d}{extension}"
        return str(directory / versioned_name)
    
    @staticmethod
    def create_output_filename(base_name: str, feature_type: str, vault_path: str) -> str:
        """
        Create a unique output filename for a specific feature
        
        Args:
            base_name: Base filename (can be with or without extension)
            feature_type: Type of feature (ascii, flowchart, analysis, etc.)
            vault_path: Vault directory path
            
        Returns:
            Full path to unique output file
        """
        # Ensure we have a clean base name
        base_path = Path(base_name)
        if base_path.suffix:
            stem = base_path.stem
            extension = base_path.suffix
        else:
            stem = str(base_path)
            extension = ".md"  # Default to markdown
        
        # Add timestamp and feature type
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create filename with feature type and timestamp
        filename = f"{stem}_{feature_type}_{timestamp}{extension}"
        
        # Full path
        vault_dir = Path(vault_path)
        full_path = vault_dir / filename
        
        # Ensure it's unique (just in case)
        if full_path.exists():
            counter = 1
            while full_path.exists():
                filename = f"{stem}_{feature_type}_{timestamp}_{counter:02d}{extension}"
                full_path = vault_dir / filename
                counter += 1
        
        return str(full_path)
    
    @staticmethod
    def backup_existing_file(file_path: str) -> Optional[str]:
        """
        Create a backup of an existing file before overwriting
        
        Args:
            file_path: Path to file to backup
            
        Returns:
            Path to backup file, or None if no backup was created
        """
        path = Path(file_path)
        
        if not path.exists():
            return None
        
        # Create backup filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{path.stem}_backup_{timestamp}{path.suffix}"
        backup_path = path.parent / backup_name
        
        try:
            # Copy file to backup location
            import shutil
            shutil.copy2(file_path, str(backup_path))
            return str(backup_path)
        except Exception as e:
            print(f"Warning: Could not create backup of {file_path}: {e}")
            return None
    
    @staticmethod
    def get_safe_filename(suggested_name: str, output_dir: str, feature_prefix: str = "") -> str:
        """
        Get a safe filename that won't overwrite existing files
        
        Args:
            suggested_name: User's suggested filename
            output_dir: Directory where file will be created
            feature_prefix: Prefix to identify feature type
            
        Returns:
            Safe filename path
        """
        # Clean the suggested name
        clean_name = re.sub(r'[<>:"/\\|?*]', '_', suggested_name)
        
        # Ensure it has an extension
        if not Path(clean_name).suffix:
            clean_name += ".md"
        
        # Add feature prefix if provided
        if feature_prefix:
            path = Path(clean_name)
            clean_name = f"{feature_prefix}_{path.stem}{path.suffix}"
        
        # Get versioned path
        full_path = Path(output_dir) / clean_name
        
        return FileVersioning.get_next_version_path(str(full_path))
    
    @staticmethod
    def list_related_files(file_path: str, feature_type: str = None) -> list:
        """
        List files related to the given file (versions, backups, etc.)
        
        Args:
            file_path: Base file path
            feature_type: Optional feature type to filter by
            
        Returns:
            List of related file paths
        """
        path = Path(file_path)
        directory = path.parent
        stem = path.stem
        extension = path.suffix
        
        # Patterns to look for
        patterns = [
            f"{stem}_v*{extension}",              # Versions
            f"{stem}_backup_*{extension}",        # Backups
        ]
        
        if feature_type:
            patterns.append(f"{stem}_{feature_type}_*{extension}")  # Feature-specific files
        
        related_files = []
        
        for pattern in patterns:
            related_files.extend(directory.glob(pattern))
        
        # Sort by modification time (newest first)
        related_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        return [str(f) for f in related_files]

# Example usage
if __name__ == "__main__":
    versioning = FileVersioning()
    
    # Test versioning
    test_file = "/tmp/test_document.md"
    
    print("Testing file versioning...")
    
    # Create some test files
    Path(test_file).touch()
    Path("/tmp/test_document_v01.md").touch()
    Path("/tmp/test_document_v02.md").touch()
    
    # Test getting next version
    next_version = versioning.get_next_version_path(test_file)
    print(f"Next version: {next_version}")
    
    # Test creating output filename
    output_file = versioning.create_output_filename("my_analysis", "flowchart", "/tmp")
    print(f"Output file: {output_file}")
    
    # Test safe filename
    safe_file = versioning.get_safe_filename("My Report", "/tmp", "analysis")
    print(f"Safe filename: {safe_file}")
    
    # Cleanup
    for f in [test_file, "/tmp/test_document_v01.md", "/tmp/test_document_v02.md"]:
        try:
            os.remove(f)
        except:
            pass