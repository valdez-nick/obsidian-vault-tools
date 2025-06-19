#!/usr/bin/env python3
"""
Audio Diagnostic Tool
Debug audio playback issues in the Enhanced Vault Manager
"""

import sys
import time
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

try:
    import pygame
    print(f"✅ Pygame version: {pygame.version.ver}")
except ImportError:
    print("❌ Pygame not available")
    sys.exit(1)

try:
    import numpy as np
    print(f"✅ NumPy available")
except ImportError:
    print("❌ NumPy not available")
    sys.exit(1)

def test_pygame_mixer():
    """Test pygame mixer initialization"""
    print("\n🔧 Testing pygame mixer...")
    
    try:
        pygame.mixer.pre_init(
            frequency=22050,
            size=-16,
            channels=2,
            buffer=512
        )
        pygame.mixer.init()
        print(f"✅ Mixer initialized successfully")
        print(f"   Frequency: {pygame.mixer.get_init()[0]}")
        print(f"   Format: {pygame.mixer.get_init()[1]}")
        print(f"   Channels: {pygame.mixer.get_init()[2]}")
        return True
    except Exception as e:
        print(f"❌ Mixer initialization failed: {e}")
        return False

def test_sound_file_loading():
    """Test loading individual sound files"""
    print("\n📁 Testing sound file loading...")
    
    sound_files = [
        'audio/sounds/effects/menu_blip.wav',
        'audio/sounds/effects/menu_select.wav',
        'audio/sounds/effects/wizard_hello.wav',
        'audio/sounds/ambient/dungeon_base.wav'
    ]
    
    loaded_sounds = {}
    
    for sound_file in sound_files:
        try:
            if Path(sound_file).exists():
                sound = pygame.mixer.Sound(sound_file)
                loaded_sounds[sound_file] = sound
                print(f"✅ Loaded: {sound_file}")
                print(f"   Length: {sound.get_length():.2f} seconds")
            else:
                print(f"❌ File not found: {sound_file}")
        except Exception as e:
            print(f"❌ Failed to load {sound_file}: {e}")
    
    return loaded_sounds

def test_direct_playback(loaded_sounds):
    """Test direct sound playback"""
    print("\n🔊 Testing direct sound playback...")
    
    for sound_file, sound in loaded_sounds.items():
        try:
            print(f"Playing: {Path(sound_file).name}")
            sound.set_volume(1.0)  # Max volume for testing
            sound.play()
            time.sleep(2)  # Wait longer to ensure we hear it
            print(f"✅ Played successfully")
        except Exception as e:
            print(f"❌ Playback failed: {e}")

def test_audio_manager():
    """Test our audio manager"""
    print("\n🎵 Testing audio manager...")
    
    try:
        from audio.audio_manager import get_audio_manager, play_sound
        
        audio_manager = get_audio_manager()
        if audio_manager and audio_manager.is_available():
            print("✅ Audio manager initialized")
            
            # Test volume levels
            print(f"   Master volume: {audio_manager.get_volume('master')}")
            print(f"   Effects volume: {audio_manager.get_volume('effects')}")
            
            # Test individual sound playback
            print("\n🎮 Testing sound effects through manager...")
            test_sounds = ['menu_blip', 'menu_select', 'wizard_hello']
            
            for sound_name in test_sounds:
                print(f"Playing: {sound_name}")
                result = play_sound(sound_name, volume=1.0)  # Max volume
                print(f"   Result: {'✅ Success' if result else '❌ Failed'}")
                time.sleep(2)
                
        else:
            print("❌ Audio manager not available")
            
    except Exception as e:
        print(f"❌ Audio manager test failed: {e}")

def test_generated_sounds():
    """Test procedural sound generation"""
    print("\n🎛️ Testing procedural sound generation...")
    
    try:
        from audio.chiptune_generator import ChiptuneGenerator
        
        generator = ChiptuneGenerator()
        if generator.available:
            print("✅ Chiptune generator available")
            
            # Generate a simple test sound
            print("Generating test square wave...")
            test_wave = generator.generate_square_wave(440, 0.5, volume=1.0)  # A4 note
            
            if test_wave is not None:
                print("✅ Sound generation successful")
                
                # Try to play the generated sound
                if pygame.mixer.get_init():
                    # Convert to stereo for pygame
                    if len(test_wave.shape) == 1:
                        stereo_wave = np.column_stack((test_wave, test_wave))
                    else:
                        stereo_wave = test_wave
                    
                    sound = pygame.sndarray.make_sound(stereo_wave)
                    print("Playing generated sound...")
                    sound.set_volume(1.0)
                    sound.play()
                    time.sleep(1)
                    print("✅ Generated sound played")
            else:
                print("❌ Sound generation failed")
        else:
            print("❌ Chiptune generator not available")
            
    except Exception as e:
        print(f"❌ Sound generation test failed: {e}")

def main():
    """Run complete audio diagnostic"""
    print("🔍 Enhanced Vault Manager - Audio Diagnostic Tool")
    print("=" * 55)
    
    # Test pygame mixer
    if not test_pygame_mixer():
        print("\n❌ Cannot continue - mixer initialization failed")
        return
    
    # Test sound file loading
    loaded_sounds = test_sound_file_loading()
    
    if not loaded_sounds:
        print("\n❌ No sound files could be loaded")
        return
    
    # Test direct playback
    test_direct_playback(loaded_sounds)
    
    # Test audio manager
    test_audio_manager()
    
    # Test sound generation
    test_generated_sounds()
    
    print("\n" + "=" * 55)
    print("🔍 Diagnostic complete!")
    print("\nIf you only heard ambient/baseline sounds:")
    print("1. Check your system audio output device")
    print("2. Try increasing system volume") 
    print("3. Check if other applications can play different audio frequencies")
    print("4. Some systems may have issues with specific WAV formats")
    
    # Cleanup
    pygame.mixer.quit()

if __name__ == "__main__":
    main()