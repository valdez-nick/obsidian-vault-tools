#!/usr/bin/env python3
"""
Isolated Audio Test
Test individual sound effects without ambient background
"""

import sys
import time
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from audio.audio_manager import get_audio_manager, play_sound
    from audio.audio_manager import start_dungeon_ambiance, stop_dungeon_ambiance
    AUDIO_AVAILABLE = True
except ImportError as e:
    print(f"Audio system not available: {e}")
    sys.exit(1)

def main():
    """Test individual sounds in isolation"""
    print("🔊 Isolated Audio Test - Individual Sound Effects")
    print("=" * 50)
    
    # Initialize audio but NO ambient
    audio_manager = get_audio_manager()
    
    if not audio_manager or not audio_manager.is_available():
        print("❌ Audio system not available")
        return
    
    print("✅ Audio system ready - NO ambient background")
    print("🔇 Ensuring no ambient sounds are playing...")
    stop_dungeon_ambiance()
    time.sleep(1)
    
    # Set volumes higher for testing
    audio_manager.set_volume('master', 1.0)
    audio_manager.set_volume('effects', 1.0)
    print("🔊 Volume set to maximum for testing")
    
    # Test each effect individually with pauses
    effects_to_test = [
        ('Menu Blip (short beep)', 'menu_blip'),
        ('Menu Select (ascending notes)', 'menu_select'),
        ('Menu Back (descending notes)', 'menu_back'),
        ('Error Chord (dissonant)', 'error_chord'),
        ('Wizard Greeting (magical arpeggio)', 'wizard_hello'),
        ('Spell Cast (frequency sweep)', 'spell_cast'),
        ('Magic Success (triumphant chord)', 'magic_success'),
        ('Wizard Warning (ominous)', 'wizard_warn'),
        ('Scan Begin (rising sweep)', 'scan_begin'),
        ('Scan Complete (success chord)', 'scan_complete'),
        ('Door Creak (slow bend)', 'door_creak'),
        ('Stone Step (percussive)', 'stone_step'),
        ('Danger Sting (sharp warning)', 'danger_sting'),
        ('Item Found (discovery bells)', 'item_found')
    ]
    
    print(f"\n🎵 Testing {len(effects_to_test)} individual sound effects...")
    print("(Each sound will play with a 3-second pause)\n")
    
    for i, (description, effect_name) in enumerate(effects_to_test, 1):
        print(f"{i:2d}. {description}")
        print(f"    Playing: {effect_name}")
        
        # Stop any currently playing sounds
        # pygame.mixer.stop()  # Stop all channels
        
        # Play the effect
        success = play_sound(effect_name, volume=1.0)
        
        if success:
            print(f"    ✅ Command sent successfully")
        else:
            print(f"    ❌ Failed to play")
        
        # Wait long enough to hear the sound
        print(f"    ⏳ Waiting 3 seconds...")
        time.sleep(3)
        print()
    
    print("🎵 Testing procedural generation in isolation...")
    
    # Test a simple procedural sound
    try:
        from audio.chiptune_generator import ChiptuneGenerator
        import pygame
        import numpy as np
        
        generator = ChiptuneGenerator()
        
        # Generate a clear, loud test tone
        print("Generating 440Hz square wave (A4 note) for 1 second...")
        test_wave = generator.generate_square_wave(440, 1.0, duty_cycle=0.5, volume=0.8)
        
        if test_wave is not None:
            # Convert to stereo
            stereo_wave = np.column_stack((test_wave, test_wave))
            
            # Create and play the sound
            sound = pygame.sndarray.make_sound(stereo_wave)
            sound.set_volume(1.0)
            print("🎵 Playing generated 440Hz tone...")
            sound.play()
            time.sleep(2)
            print("✅ Generated sound test complete")
        else:
            print("❌ Failed to generate test sound")
            
    except Exception as e:
        print(f"❌ Procedural test failed: {e}")
    
    # Test ambient separately
    print("\n🏰 Now testing ambient track in isolation...")
    print("(This should be the low droning sound you mentioned)")
    start_dungeon_ambiance()
    print("🎵 Ambient playing for 5 seconds...")
    time.sleep(5)
    stop_dungeon_ambiance()
    print("🔇 Ambient stopped")
    
    print("\n" + "=" * 50)
    print("🔍 Isolated test complete!")
    print("\nDid you hear distinct differences between the sound effects?")
    print("- Menu sounds should be short blips and beeps")
    print("- Wizard sounds should have magical, sweeping qualities") 
    print("- Error sounds should be harsh/dissonant")
    print("- The 440Hz test tone should be a clear, steady beep")
    print("\nIf all sounds seem similar, there may be:")
    print("1. Audio driver issues with frequency response")
    print("2. System audio processing affecting the output")
    print("3. Hardware limitations in your audio output device")
    
    # Cleanup
    if audio_manager:
        audio_manager.cleanup()

if __name__ == "__main__":
    main()