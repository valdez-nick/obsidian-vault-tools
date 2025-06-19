#!/usr/bin/env python3
"""
Test Obvious Sounds
Test the new, very audible sound effects
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    sys.exit(1)

def test_obvious_sounds():
    """Test the new obvious sound effects"""
    print("🔊 Testing New OBVIOUS Sound Effects")
    print("=" * 40)
    
    # Initialize pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
    print(f"✅ Audio initialized: {pygame.mixer.get_init()}")
    
    # Load and test each obvious sound
    obvious_sounds = [
        ("Short Loud Beep", "menu_blip_obvious.wav"),
        ("Ascending Beeps", "menu_select_obvious.wav"),
        ("Descending Beeps", "menu_back_obvious.wav"),
        ("Dissonant Error", "error_obvious.wav"),
        ("Frequency Sweep", "sweep_obvious.wav"),
        ("Multi-Tone Greeting", "wizard_hello_obvious.wav")
    ]
    
    print("\n🎵 Playing each sound with 3-second pauses...")
    print("(These should be VERY audible - loud, clear tones)")
    
    for i, (description, filename) in enumerate(obvious_sounds, 1):
        print(f"\n{i}. {description}")
        
        sound_path = f"audio/sounds/effects/{filename}"
        
        if Path(sound_path).exists():
            try:
                # Load and play with pygame
                sound = pygame.mixer.Sound(sound_path)
                sound.set_volume(1.0)  # Maximum volume
                
                print(f"   🔊 Playing: {filename}")
                sound.play()
                
                # Wait for sound to finish
                time.sleep(3)
                
                print(f"   ✅ Completed")
                
            except Exception as e:
                print(f"   ❌ Failed to play {filename}: {e}")
        else:
            print(f"   ❌ File not found: {sound_path}")
    
    print("\n" + "=" * 40)
    print("🔊 Obvious sound test complete!")
    print("\nIf you heard these sounds clearly:")
    print("✓ The issue was with the original 8-bit sounds being too subtle")
    print("✓ We can update the audio system to use these clearer sounds")
    print("\nIf you still only hear hums:")
    print("❌ There may be a deeper audio system or hardware issue")
    
    pygame.mixer.quit()

def main():
    test_obvious_sounds()

if __name__ == "__main__":
    main()