#!/usr/bin/env python3
"""
Improved Audio Test
Test the enhanced audio system with better channel separation
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
    sys.exit(1)

def main():
    """Test the improved audio system with better separation"""
    print("🎵 Enhanced Audio System Test - Improved Channel Separation")
    print("=" * 60)
    
    # Initialize audio
    audio_manager = get_audio_manager()
    
    if not audio_manager or not audio_manager.is_available():
        print("❌ Audio system not available")
        return
    
    print("✅ Enhanced audio system initialized")
    
    # Set clear volumes for testing
    audio_manager.set_volume('master', 0.9)
    audio_manager.set_volume('effects', 1.0)  # Full effects volume
    audio_manager.set_volume('ambient', 0.2)  # Lower ambient so effects are clearer
    
    print("🔊 Volume settings optimized for clear effect separation:")
    print(f"   Master: {audio_manager.get_volume('master'):.1f}")
    print(f"   Effects: {audio_manager.get_volume('effects'):.1f}")
    print(f"   Ambient: {audio_manager.get_volume('ambient'):.1f}")
    
    # Test 1: Effects only (no ambient)
    print("\n🎮 TEST 1: Sound effects only (no ambient background)")
    print("-" * 40)
    
    stop_dungeon_ambiance()  # Ensure no ambient
    time.sleep(0.5)
    
    clear_effects = [
        ('Sharp Menu Blip', 'menu_blip'),
        ('Ascending Notes', 'menu_select'),
        ('Descending Notes', 'menu_back'),
        ('Dissonant Error', 'error_chord'),
        ('Magical Greeting', 'wizard_hello')
    ]
    
    for description, effect in clear_effects:
        print(f"Playing: {description}")
        play_sound(effect, volume=1.0)  # Max volume
        time.sleep(2.5)  # Longer pause for clarity
    
    # Test 2: With low ambient background
    print("\n🏰 TEST 2: Effects with low ambient background")
    print("-" * 40)
    
    print("Starting low-volume ambient...")
    start_dungeon_ambiance()
    time.sleep(1)
    
    dynamic_effects = [
        ('Wizard Spell Cast', 'spell_cast'),
        ('Magic Success', 'magic_success'),
        ('Danger Warning', 'danger_sting'),
        ('Door Creak', 'door_creak'),
        ('Stone Step', 'stone_step')
    ]
    
    for description, effect in dynamic_effects:
        print(f"Playing: {description} (over ambient)")
        play_sound(effect, volume=1.0)
        time.sleep(2.5)
    
    # Test 3: Rapid menu simulation
    print("\n⚡ TEST 3: Rapid menu navigation simulation")
    print("-" * 40)
    
    stop_dungeon_ambiance()
    time.sleep(0.5)
    
    print("Simulating quick menu navigation...")
    menu_sequence = [
        'menu_blip',    # Navigate
        'menu_blip',    # Navigate  
        'menu_select',  # Select
        'menu_blip',    # Navigate
        'menu_blip',    # Navigate
        'menu_select',  # Select
        'menu_back',    # Back
        'menu_back'     # Back to main
    ]
    
    for i, effect in enumerate(menu_sequence, 1):
        action = "Navigate" if effect == 'menu_blip' else ("Select" if effect == 'menu_select' else "Back")
        print(f"  {i}. {action}")
        play_sound(effect)
        time.sleep(0.8)  # Quick menu timing
    
    # Test 4: High-contrast frequency test
    print("\n🎼 TEST 4: High-contrast frequency demonstration")
    print("-" * 40)
    
    try:
        from audio.chiptune_generator import ChiptuneGenerator
        import pygame
        import numpy as np
        
        generator = ChiptuneGenerator()
        
        # Generate contrasting tones
        tones = [
            (220, "Low A (220Hz)"),
            (440, "Middle A (440Hz)"),
            (880, "High A (880Hz)"),
            (1760, "Very High A (1760Hz)")
        ]
        
        for freq, description in tones:
            print(f"Playing: {description}")
            
            # Generate pure square wave
            test_wave = generator.generate_square_wave(freq, 0.8, duty_cycle=0.5, volume=0.7)
            
            if test_wave is not None:
                # Convert to stereo
                stereo_wave = np.column_stack((test_wave, test_wave))
                
                # Create and play
                sound = pygame.sndarray.make_sound(stereo_wave)
                sound.set_volume(1.0)
                sound.play()
                time.sleep(1.2)
            
    except Exception as e:
        print(f"Frequency test failed: {e}")
    
    # Test 5: Full experience simulation
    print("\n🧙‍♂️ TEST 5: Full enhanced vault manager experience")
    print("-" * 40)
    
    print("Starting full dungeon crawler experience...")
    
    # Wizard greeting
    print("1. Wizard Welcome")
    wizard_greeting()
    time.sleep(3)
    
    # Start ambient
    print("2. Dungeon Atmosphere")
    start_dungeon_ambiance()
    time.sleep(2)
    
    # Menu navigation
    print("3. Menu Navigation")
    play_sound('menu_blip')
    time.sleep(1)
    play_sound('menu_select')
    time.sleep(2)
    
    # ASCII operation
    print("4. ASCII Art Generation")
    play_sound('magic_begin')
    time.sleep(1)
    play_sound('magic_complete')
    time.sleep(1)
    magical_success()
    time.sleep(2)
    
    # Warning example
    print("5. Error Handling")
    play_sound('error_chord')
    time.sleep(1)
    dungeon_danger()
    time.sleep(2)
    
    # Cleanup
    print("6. Cleanup")
    stop_dungeon_ambiance()
    time.sleep(1)
    
    print("\n" + "=" * 60)
    print("🎵 Enhanced audio test complete!")
    print("\nWith the improved audio system you should now hear:")
    print("✓ Clear separation between ambient and effects")
    print("✓ No overlapping sounds interfering with each other")  
    print("✓ Distinct character for each type of sound effect")
    print("✓ Proper volume balance between background and foreground")
    
    # Cleanup
    if audio_manager:
        audio_manager.cleanup()

if __name__ == "__main__":
    main()