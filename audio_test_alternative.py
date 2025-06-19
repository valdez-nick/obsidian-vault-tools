#!/usr/bin/env python3
"""
Alternative Audio Test
Test using different pygame settings and direct system audio
"""

import sys
import time
import subprocess
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

try:
    import pygame
    import numpy as np
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

def test_system_audio():
    """Test system audio capabilities"""
    print("🔊 Testing system audio capabilities...")
    
    # Try to play a simple system beep
    try:
        if sys.platform == 'darwin':  # macOS
            print("Testing macOS system beep...")
            subprocess.run(['afplay', '/System/Library/Sounds/Ping.aiff'], check=True)
            print("✅ System beep worked")
        elif sys.platform == 'linux':
            print("Testing Linux system beep...")
            subprocess.run(['paplay', '/usr/share/sounds/alsa/Front_Left.wav'], check=False)
        elif sys.platform == 'win32':
            print("Testing Windows system beep...")
            import winsound
            winsound.Beep(440, 1000)
    except Exception as e:
        print(f"❌ System audio test failed: {e}")

def test_pygame_different_settings():
    """Test pygame with different initialization settings"""
    print("\n🎮 Testing pygame with different audio settings...")
    
    if not PYGAME_AVAILABLE:
        print("❌ Pygame not available")
        return
    
    # Test different configurations
    configs = [
        {"frequency": 44100, "size": -16, "channels": 2, "buffer": 1024},
        {"frequency": 22050, "size": -16, "channels": 1, "buffer": 512},
        {"frequency": 44100, "size": 16, "channels": 2, "buffer": 2048},
        {"frequency": 11025, "size": -8, "channels": 1, "buffer": 256},
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\n{i}. Testing config: {config}")
        
        try:
            # Quit any existing mixer
            if pygame.mixer.get_init():
                pygame.mixer.quit()
            
            # Initialize with new settings
            pygame.mixer.pre_init(**config)
            pygame.mixer.init()
            
            print(f"   ✅ Initialized: {pygame.mixer.get_init()}")
            
            # Generate a simple test tone
            frequency = 800  # Higher frequency for better audibility
            duration = 0.5
            sample_rate = config["frequency"]
            
            # Generate sine wave (easier to hear than square wave)
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            wave = np.sin(2 * np.pi * frequency * t)
            
            # Convert to pygame format
            if config["size"] < 0:  # Signed
                if abs(config["size"]) == 16:
                    wave = (wave * 32767).astype(np.int16)
                else:
                    wave = (wave * 127).astype(np.int8)
            else:  # Unsigned
                if config["size"] == 16:
                    wave = ((wave + 1) * 32767).astype(np.uint16)
                else:
                    wave = ((wave + 1) * 127).astype(np.uint8)
            
            # Make stereo if needed
            if config["channels"] == 2:
                if wave.ndim == 1:
                    wave = np.column_stack((wave, wave))
            
            # Create and play sound
            sound = pygame.sndarray.make_sound(wave)
            sound.set_volume(1.0)
            
            print(f"   🎵 Playing {frequency}Hz sine wave...")
            sound.play()
            time.sleep(1)
            
        except Exception as e:
            print(f"   ❌ Config {i} failed: {e}")
    
    # Cleanup
    if pygame.mixer.get_init():
        pygame.mixer.quit()

def test_direct_wav_playback():
    """Test playing WAV files directly with system tools"""
    print("\n📁 Testing direct WAV file playback with system tools...")
    
    test_files = [
        'audio/sounds/effects/menu_blip.wav',
        'audio/sounds/effects/wizard_hello.wav'
    ]
    
    for wav_file in test_files:
        if Path(wav_file).exists():
            print(f"Playing {wav_file} with system player...")
            try:
                if sys.platform == 'darwin':  # macOS
                    subprocess.run(['afplay', wav_file], check=True)
                elif sys.platform == 'linux':
                    subprocess.run(['paplay', wav_file], check=False)
                elif sys.platform == 'win32':
                    subprocess.run(['start', wav_file], shell=True, check=False)
                print(f"✅ Played {wav_file}")
            except Exception as e:
                print(f"❌ Failed to play {wav_file}: {e}")
        else:
            print(f"❌ File not found: {wav_file}")

def test_frequency_range():
    """Test different frequency ranges to check audio system response"""
    print("\n🎼 Testing frequency range response...")
    
    if not PYGAME_AVAILABLE:
        print("❌ Pygame not available")
        return
    
    try:
        # Initialize with standard settings
        pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=1024)
        pygame.mixer.init()
        
        frequencies = [100, 200, 440, 800, 1600, 3200]  # Range from low to high
        
        for freq in frequencies:
            print(f"Testing {freq}Hz...")
            
            # Generate pure tone
            duration = 1.0
            sample_rate = 44100
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            
            # Use sine wave for cleaner sound
            wave = np.sin(2 * np.pi * freq * t)
            wave = (wave * 0.5 * 32767).astype(np.int16)  # 50% volume
            
            # Make stereo
            stereo_wave = np.column_stack((wave, wave))
            
            # Play
            sound = pygame.sndarray.make_sound(stereo_wave)
            sound.set_volume(1.0)
            sound.play()
            time.sleep(1.5)
            
        print("✅ Frequency range test complete")
        
    except Exception as e:
        print(f"❌ Frequency test failed: {e}")
    finally:
        if pygame.mixer.get_init():
            pygame.mixer.quit()

def main():
    """Run comprehensive audio testing with alternatives"""
    print("🔍 Alternative Audio Testing Suite")
    print("=" * 40)
    
    # Test system audio first
    test_system_audio()
    
    # Test pygame with different settings
    test_pygame_different_settings()
    
    # Test direct WAV playback
    test_direct_wav_playback()
    
    # Test frequency range
    test_frequency_range()
    
    print("\n" + "=" * 40)
    print("🔍 Alternative testing complete!")
    print("\nThis test should help identify:")
    print("1. Whether your system can play audio at all")
    print("2. Which pygame settings work best")
    print("3. Whether the WAV files are valid")
    print("4. Which frequency ranges your system can reproduce")

if __name__ == "__main__":
    main()