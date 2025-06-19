#!/usr/bin/env python3
"""
Regenerate Obvious Sounds
Create very audible, clear sound effects that should be impossible to miss
"""

import sys
import time
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent))

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    sys.exit(1)

class ObviousSoundGenerator:
    """Generate very obvious, audible sound effects"""
    
    def __init__(self):
        self.sample_rate = 44100
    
    def create_loud_beep(self, frequency=800, duration=0.3, volume=0.8):
        """Create a loud, obvious beep"""
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # Use sine wave - easier to hear than square wave
        wave = volume * np.sin(2 * np.pi * frequency * t)
        
        # Add slight attack/decay to prevent clicks
        attack_samples = int(0.01 * self.sample_rate)  # 10ms attack
        decay_samples = int(0.05 * self.sample_rate)   # 50ms decay
        
        # Attack
        wave[:attack_samples] *= np.linspace(0, 1, attack_samples)
        
        # Decay
        wave[-decay_samples:] *= np.linspace(1, 0, decay_samples)
        
        # Convert to 16-bit and make stereo
        wave_int = (wave * 32767).astype(np.int16)
        stereo_wave = np.column_stack((wave_int, wave_int))
        
        return stereo_wave
    
    def create_ascending_beeps(self, base_freq=400, duration=0.15, count=3):
        """Create ascending beeps"""
        waves = []
        
        for i in range(count):
            freq = base_freq * (1.5 ** i)  # Each beep 1.5x higher
            beep = self.create_loud_beep(freq, duration, 0.7)
            waves.append(beep)
            
            # Add small gap between beeps
            if i < count - 1:
                gap_samples = int(0.05 * self.sample_rate)  # 50ms gap
                gap = np.zeros((gap_samples, 2), dtype=np.int16)
                waves.append(gap)
        
        return np.concatenate(waves)
    
    def create_descending_beeps(self, base_freq=800, duration=0.15, count=3):
        """Create descending beeps"""
        waves = []
        
        for i in range(count):
            freq = base_freq / (1.5 ** i)  # Each beep 1.5x lower
            beep = self.create_loud_beep(freq, duration, 0.7)
            waves.append(beep)
            
            # Add small gap between beeps
            if i < count - 1:
                gap_samples = int(0.05 * self.sample_rate)
                gap = np.zeros((gap_samples, 2), dtype=np.int16)
                waves.append(gap)
        
        return np.concatenate(waves)
    
    def create_error_sound(self):
        """Create harsh, dissonant error sound"""
        duration = 0.4
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # Mix two dissonant frequencies
        freq1 = 200
        freq2 = 300  # Creates beating/dissonance
        
        wave1 = 0.4 * np.sin(2 * np.pi * freq1 * t)
        wave2 = 0.4 * np.sin(2 * np.pi * freq2 * t)
        
        # Combine waves
        combined = wave1 + wave2
        
        # Add envelope
        attack_samples = int(0.02 * self.sample_rate)
        decay_samples = int(0.1 * self.sample_rate)
        
        combined[:attack_samples] *= np.linspace(0, 1, attack_samples)
        combined[-decay_samples:] *= np.linspace(1, 0, decay_samples)
        
        # Convert and make stereo
        wave_int = (combined * 32767).astype(np.int16)
        stereo_wave = np.column_stack((wave_int, wave_int))
        
        return stereo_wave
    
    def create_sweep_sound(self, start_freq=200, end_freq=1000, duration=0.8):
        """Create frequency sweep"""
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # Logarithmic frequency sweep
        freq = start_freq * (end_freq / start_freq) ** (t / duration)
        
        # Generate wave
        phase = np.cumsum(2 * np.pi * freq / self.sample_rate)
        wave = 0.6 * np.sin(phase)
        
        # Envelope
        attack_samples = int(0.1 * self.sample_rate)
        release_samples = int(0.2 * self.sample_rate)
        
        wave[:attack_samples] *= np.linspace(0, 1, attack_samples)
        wave[-release_samples:] *= np.linspace(1, 0, release_samples)
        
        # Convert and make stereo
        wave_int = (wave * 32767).astype(np.int16)
        stereo_wave = np.column_stack((wave_int, wave_int))
        
        return stereo_wave
    
    def save_wav(self, audio_data, filename):
        """Save audio data as WAV file"""
        import wave
        
        filepath = Path(f"audio/sounds/effects/{filename}.wav")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with wave.open(str(filepath), 'wb') as wav_file:
            wav_file.setnchannels(2)  # Stereo
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        print(f"✅ Created {filepath}")
        return filepath

def main():
    """Generate new, obvious sound effects"""
    print("🔊 Regenerating OBVIOUS Sound Effects")
    print("=" * 40)
    
    generator = ObviousSoundGenerator()
    
    # Create very obvious versions of key sounds
    sounds_to_create = [
        ("menu_blip_obvious", lambda: generator.create_loud_beep(800, 0.2, 0.9)),
        ("menu_select_obvious", lambda: generator.create_ascending_beeps(400, 0.2, 3)),
        ("menu_back_obvious", lambda: generator.create_descending_beeps(800, 0.2, 3)),
        ("error_obvious", lambda: generator.create_error_sound()),
        ("sweep_obvious", lambda: generator.create_sweep_sound(300, 1200, 0.6)),
        ("wizard_hello_obvious", lambda: generator.create_ascending_beeps(300, 0.25, 5))
    ]
    
    created_files = []
    
    for name, generator_func in sounds_to_create:
        print(f"Creating {name}...")
        audio_data = generator_func()
        filepath = generator.save_wav(audio_data, name)
        created_files.append(filepath)
    
    # Test each sound immediately after creation
    print(f"\n🎵 Testing created sounds...")
    
    for filepath in created_files:
        print(f"Playing: {filepath.name}")
        
        try:
            import subprocess
            subprocess.run(['afplay', str(filepath)], check=True, timeout=5)
            time.sleep(0.5)  # Brief pause between sounds
        except Exception as e:
            print(f"❌ Failed to play {filepath}: {e}")
    
    print(f"\n✅ Created {len(created_files)} obvious sound effects")
    print("\nThese sounds use:")
    print("- Clear sine waves (not square waves)")
    print("- Higher frequencies (400-1200Hz)")
    print("- Longer durations (0.2-0.8 seconds)")
    print("- Maximum volume levels")
    print("- Proper envelopes to prevent audio artifacts")
    
    # Now test with pygame
    print(f"\n🎮 Testing with pygame...")
    
    try:
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
        
        for filepath in created_files[:3]:  # Test first 3
            print(f"Pygame test: {filepath.name}")
            sound = pygame.mixer.Sound(str(filepath))
            sound.set_volume(1.0)
            sound.play()
            time.sleep(2)
        
        pygame.mixer.quit()
        print("✅ Pygame tests complete")
        
    except Exception as e:
        print(f"❌ Pygame test failed: {e}")

if __name__ == "__main__":
    main()