#!/usr/bin/env python3
"""
Quick Audio Fix
Replace the subtle 8-bit sounds with obvious, audible ones in the audio manager
"""

import sys
from pathlib import Path
import shutil

def backup_original_sounds():
    """Backup original sound files"""
    print("📦 Backing up original sound files...")
    
    effects_dir = Path("audio/sounds/effects")
    backup_dir = Path("audio/sounds/effects_backup")
    backup_dir.mkdir(exist_ok=True)
    
    original_sounds = [
        "menu_blip.wav",
        "menu_select.wav", 
        "menu_back.wav",
        "error_chord.wav",
        "wizard_hello.wav"
    ]
    
    for sound in original_sounds:
        original_path = effects_dir / sound
        backup_path = backup_dir / sound
        
        if original_path.exists():
            shutil.copy2(original_path, backup_path)
            print(f"✅ Backed up {sound}")

def replace_with_obvious_sounds():
    """Replace original sounds with obvious versions"""
    print("\n🔄 Replacing with obvious sounds...")
    
    effects_dir = Path("audio/sounds/effects")
    
    replacements = [
        ("menu_blip_obvious.wav", "menu_blip.wav"),
        ("menu_select_obvious.wav", "menu_select.wav"),
        ("menu_back_obvious.wav", "menu_back.wav"),
        ("error_obvious.wav", "error_chord.wav"),
        ("wizard_hello_obvious.wav", "wizard_hello.wav"),
        ("sweep_obvious.wav", "spell_cast.wav"),  # Replace spell cast with sweep
    ]
    
    for obvious_name, target_name in replacements:
        obvious_path = effects_dir / obvious_name
        target_path = effects_dir / target_name
        
        if obvious_path.exists():
            # Copy obvious sound to replace original
            shutil.copy2(obvious_path, target_path)
            print(f"✅ Replaced {target_name} with {obvious_name}")
        else:
            print(f"❌ {obvious_name} not found")

def test_vault_manager_with_obvious_sounds():
    """Test just the audio parts of vault manager"""
    print("\n🧙‍♂️ Testing vault manager audio with obvious sounds...")
    
    try:
        sys.path.append(str(Path(__file__).parent))
        from audio.audio_manager import get_audio_manager, play_sound
        from audio.audio_manager import wizard_greeting, magical_success
        
        audio_manager = get_audio_manager()
        
        if audio_manager and audio_manager.is_available():
            print("✅ Audio manager ready")
            
            # Set high volumes
            audio_manager.set_volume('master', 1.0)
            audio_manager.set_volume('effects', 1.0)
            
            # Test key sounds that should now be obvious
            print("\n🎵 Testing replaced sounds in audio manager...")
            
            test_sounds = [
                ("Menu Navigation", "menu_blip"),
                ("Menu Selection", "menu_select"), 
                ("Menu Back", "menu_back"),
                ("Error Sound", "error_chord"),
                ("Wizard Greeting", "wizard_hello"),
                ("Spell Cast", "spell_cast")
            ]
            
            for description, sound_name in test_sounds:
                print(f"Playing: {description}")
                success = play_sound(sound_name, volume=1.0)
                print(f"   Result: {'✅ Success' if success else '❌ Failed'}")
                
                import time
                time.sleep(2)  # Pause between sounds
            
            print("\n🎯 If you heard these sounds clearly, the fix worked!")
            
        else:
            print("❌ Audio manager not available")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

def main():
    """Apply the audio fix"""
    print("🔧 Audio Fix - Replace Subtle Sounds with Obvious Ones")
    print("=" * 55)
    
    # Backup originals
    backup_original_sounds()
    
    # Replace with obvious versions
    replace_with_obvious_sounds()
    
    # Test the fix
    test_vault_manager_with_obvious_sounds()
    
    print("\n" + "=" * 55)
    print("🔧 Audio fix complete!")
    print("\nThe Enhanced Vault Manager should now use:")
    print("✓ Loud, clear beeps instead of subtle 8-bit sounds")
    print("✓ Higher frequency tones (400-1200Hz)")
    print("✓ Longer duration sounds (0.2-0.8 seconds)")
    print("✓ Maximum volume levels")
    print("\nTo test: run ./obsidian_manager_enhanced")
    print("To restore originals: copy files from audio/sounds/effects_backup/")

if __name__ == "__main__":
    main()