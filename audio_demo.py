#!/usr/bin/env python3
"""
Audio Demo for Enhanced Vault Manager
Demonstrates the 8-bit dungeon crawler sound system
"""

import sys
import time
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from audio.audio_manager import get_audio_manager, play_sound
    from audio.audio_manager import start_dungeon_ambiance, stop_dungeon_ambiance
    from audio.audio_manager import wizard_greeting, magical_success, dungeon_danger
    AUDIO_AVAILABLE = True
except ImportError as e:
    print(f"Audio system not available: {e}")
    AUDIO_AVAILABLE = False
    sys.exit(1)

def main():
    """Run audio demo"""
    print("🧙‍♂️ Enhanced Vault Manager - Audio System Demo")
    print("=" * 50)
    
    # Initialize audio
    audio_manager = get_audio_manager()
    
    if not audio_manager or not audio_manager.is_available():
        print("❌ Audio system not available")
        return
    
    print("✅ Audio system initialized successfully!")
    print(f"   - Master Volume: {audio_manager.get_volume('master'):.1f}")
    print(f"   - Effects Volume: {audio_manager.get_volume('effects'):.1f}")
    print(f"   - Ambient Volume: {audio_manager.get_volume('ambient'):.1f}")
    
    # Start ambient atmosphere
    print("\n🏰 Starting dungeon atmosphere...")
    start_dungeon_ambiance()
    time.sleep(1)
    
    # Test menu sounds
    print("\n🎵 Testing menu navigation sounds...")
    print("   Playing: Menu blip")
    play_sound('menu_blip')
    time.sleep(1)
    
    print("   Playing: Menu select")
    play_sound('menu_select')
    time.sleep(1.5)
    
    print("   Playing: Menu back")
    play_sound('menu_back')
    time.sleep(1.5)
    
    # Test magical sounds
    print("\n✨ Testing magical effects...")
    print("   Playing: Wizard greeting")
    wizard_greeting()
    time.sleep(2)
    
    print("   Playing: Spell casting")
    play_sound('spell_cast')
    time.sleep(1.5)
    
    print("   Playing: Magic success")
    magical_success()
    time.sleep(2)
    
    # Test vault operation sounds
    print("\n📊 Testing vault operation sounds...")
    print("   Playing: Scan begin")
    play_sound('scan_begin')
    time.sleep(1.5)
    
    print("   Playing: Scan complete")
    play_sound('scan_complete')
    time.sleep(1.5)
    
    # Test warning sounds
    print("\n⚠️ Testing warning sounds...")
    print("   Playing: Error chord")
    play_sound('error_chord')
    time.sleep(1.5)
    
    print("   Playing: Danger warning")
    dungeon_danger()
    time.sleep(2)
    
    # Test dungeon atmosphere
    print("\n🏰 Testing dungeon effects...")
    print("   Playing: Door creak")
    play_sound('door_creak')
    time.sleep(2)
    
    print("   Playing: Stone step")
    play_sound('stone_step')
    time.sleep(1)
    
    print("   Playing: Item found")
    play_sound('item_found')
    time.sleep(2)
    
    # Cleanup
    print("\n🎵 Demo complete! Cleaning up...")
    stop_dungeon_ambiance()
    audio_manager.cleanup()
    
    print("\n" + "=" * 50)
    print("🧙‍♂️ The digital wizard has demonstrated the audio system!")
    print("Your Enhanced Vault Manager now has full 8-bit dungeon crawler atmosphere! 🏰🎵")

if __name__ == "__main__":
    main()