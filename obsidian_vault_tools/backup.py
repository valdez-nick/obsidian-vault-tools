"""
Vault backup functionality
"""

import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

class BackupManager:
    """Manages vault backups"""
    
    def __init__(self, vault_path: str):
        self.vault_path = vault_path
        self.backup_dir = os.path.join(os.path.dirname(vault_path), "vault-backups")
        os.makedirs(self.backup_dir, exist_ok=True)
    
    def create_backup(self, backup_name: Optional[str] = None) -> Dict[str, Any]:
        """Create a backup of the vault"""
        if not backup_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"vault_backup_{timestamp}"
        
        backup_path = os.path.join(self.backup_dir, f"{backup_name}.zip")
        
        try:
            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(self.vault_path):
                    for file in files:
                        if not file.startswith('.'):  # Skip hidden files
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, self.vault_path)
                            zipf.write(file_path, arcname)
            
            backup_size = os.path.getsize(backup_path)
            
            return {
                "success": True,
                "backup_path": backup_path,
                "backup_name": backup_name,
                "backup_size": backup_size,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def list_backups(self) -> Dict[str, Any]:
        """List all available backups"""
        backups = []
        
        if os.path.exists(self.backup_dir):
            for file in os.listdir(self.backup_dir):
                if file.endswith('.zip'):
                    file_path = os.path.join(self.backup_dir, file)
                    stat = os.stat(file_path)
                    
                    backups.append({
                        "name": file,
                        "path": file_path,
                        "size": stat.st_size,
                        "created": datetime.fromtimestamp(stat.st_ctime).isoformat()
                    })
        
        # Sort by creation time, newest first
        backups.sort(key=lambda x: x["created"], reverse=True)
        
        return {
            "total_backups": len(backups),
            "backup_directory": self.backup_dir,
            "backups": backups
        }
    
    def restore_backup(self, backup_name: str, restore_path: Optional[str] = None) -> Dict[str, Any]:
        """Restore a backup"""
        if not restore_path:
            restore_path = self.vault_path + "_restored"
        
        backup_path = os.path.join(self.backup_dir, backup_name)
        if not backup_name.endswith('.zip'):
            backup_path += '.zip'
        
        if not os.path.exists(backup_path):
            return {
                "success": False,
                "error": f"Backup not found: {backup_path}"
            }
        
        try:
            os.makedirs(restore_path, exist_ok=True)
            
            with zipfile.ZipFile(backup_path, 'r') as zipf:
                zipf.extractall(restore_path)
            
            return {
                "success": True,
                "restore_path": restore_path,
                "backup_path": backup_path,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def cleanup_old_backups(self, keep_count: int = 5) -> Dict[str, Any]:
        """Remove old backups, keeping only the most recent ones"""
        backups_info = self.list_backups()
        backups = backups_info["backups"]
        
        if len(backups) <= keep_count:
            return {
                "success": True,
                "removed_count": 0,
                "kept_count": len(backups),
                "message": "No backups removed"
            }
        
        # Remove oldest backups
        to_remove = backups[keep_count:]
        removed_count = 0
        
        for backup in to_remove:
            try:
                os.remove(backup["path"])
                removed_count += 1
            except Exception as e:
                print(f"Error removing backup {backup['name']}: {e}")
        
        return {
            "success": True,
            "removed_count": removed_count,
            "kept_count": len(backups) - removed_count,
            "message": f"Removed {removed_count} old backups"
        }