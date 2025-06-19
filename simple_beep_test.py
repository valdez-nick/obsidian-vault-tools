#!/usr/bin/env python3
"""
Simple Beep Test
Generate a very obvious, loud beep to test if basic audio generation works
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

try:
    import pygame
    import numpy as np
    PYGAME_AVAILABLE = True
except ImportError:
    print("Required libraries not available")
    sys.exit(1)

def create_obvious_beep():
    """Create a very obvious, loud beep sound"""
    print("🔊 Creating very obvious test beep...")
    
    # Audio settings
    sample_rate = 44100
    frequency = 1000  # 1kHz - very audible frequency
    duration = 1.0    # 1 second
    
    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Generate loud sine wave (easier to hear than square wave)
    amplitude = 0.8  # 80% volume
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    
    # Convert to 16-bit signed integers
    wave_int = (wave * 32767).astype(np.int16)
    
    # Make it stereo
    stereo_wave = np.column_stack((wave_int, wave_int))
    
    return stereo_wave

def test_pygame_simple():
    """Test pygame with the simplest possible setup"""
    print("🎮 Testing pygame with simplest setup...")
    
    try:
        # Initialize pygame mixer with standard settings
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
        print(f"✅ Pygame mixer initialized: {pygame.mixer.get_init()}")
        
        # Create obvious beep
        beep_data = create_obvious_beep()
        
        # Create pygame sound
        beep_sound = pygame.sndarray.make_sound(beep_data)
        beep_sound.set_volume(1.0)  # Maximum volume
        
        print("🔊 Playing 1000Hz beep for 1 second at maximum volume...")
        print("   (You should hear a clear, loud tone)")
        
        beep_sound.play()
        time.sleep(2)  # Wait for sound to finish
        
        print("✅ Beep test complete")
        
        # Test a different frequency
        print("\n🔊 Testing different frequency (500Hz)...")
        
        # Generate 500Hz beep
        t = np.linspace(0, 1.0, 44100, False)
        wave = 0.8 * np.sin(2 * np.pi * 500 * t)
        wave_int = (wave * 32767).astype(np.int16)
        stereo_wave = np.column_stack((wave_int, wave_int))
        
        beep2 = pygame.sndarray.make_sound(stereo_wave)
        beep2.set_volume(1.0)
        beep2.play()
        time.sleep(2)
        
        print("✅ Second beep test complete")
        
    except Exception as e:
        print(f"❌ Pygame test failed: {e}")
    finally:
        pygame.mixer.quit()

def test_system_beep():
    """Test system audio"""
    print("\n🖥️ Testing system beep...")
    
    try:
        import subprocess
        # macOS system beep
        subprocess.run(['osascript', '-e', 'beep 3'], check=True)
        print("✅ System beep test complete")
    except Exception as e:
        print(f"❌ System beep failed: {e}")

def save_test_wav():
    """Save a test WAV file and try to play it"""
    print("\n📁 Creating and testing WAV file...")
    
    try:
        # Create obvious beep
        beep_data = create_obvious_beep()
        
        # Save using wave module
        import wave
        
        with wave.open('test_beep.wav', 'wb') as wav_file:
            wav_file.setnchannels(2)  # Stereo
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(44100)
            wav_file.writeframes(beep_data.tobytes())
        
        print("✅ test_beep.wav created")
        
        # Try to play with system player
        import subprocess
        print("🔊 Playing test_beep.wav with system player...")
        subprocess.run(['afplay', 'test_beep.wav'], check=True)
        print("✅ WAV file playback complete")
        
        # Clean up
        Path('test_beep.wav').unlink()
        
    except Exception as e:
        print(f"❌ WAV test failed: {e}")

def main():
    """Run all basic audio tests"""
    print("🔍 Simple Beep Test - Basic Audio Diagnosis")
    print("=" * 45)
    
    # Test system audio first
    test_system_beep()
    
    # Test pygame
    test_pygame_simple()
    
    # Test WAV file creation and playback
    save_test_wav()
    
    print("\n" + "=" * 45)
    print("🔍 Basic audio tests complete!")
    print("\nIf you heard:")
    print("✓ System beeps - Your audio hardware works")
    print("✓ Clear 1000Hz and 500Hz tones - Pygame works correctly")
    print("✓ test_beep.wav - WAV generation works")
    print("\nIf you only heard hums or no sound:")
    print("❌ There may be an audio driver or format issue")

if __name__ == "__main__":
    main()