#!/usr/bin/env python3
"""
8-bit Chiptune Sound Generator
Creates authentic retro sounds using square wave synthesis and classic waveforms
Designed for creepy dungeon crawler atmosphere with wizard themes
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Union, List
import math
import random

# Try to import required libraries
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

class ChiptuneGenerator:
    """
    Generates authentic 8-bit chiptune sounds using classic waveforms
    Focuses on creepy dungeon crawler atmosphere with wizard themes
    """
    
    # Standard chiptune sample rate for authentic retro feel
    SAMPLE_RATE = 22050
    
    # Classic 8-bit frequencies (C major scale in various octaves)
    NOTES = {
        'C2': 65.41, 'D2': 73.42, 'E2': 82.41, 'F2': 87.31, 'G2': 98.00, 'A2': 110.00, 'B2': 123.47,
        'C3': 130.81, 'D3': 146.83, 'E3': 164.81, 'F3': 174.61, 'G3': 196.00, 'A3': 220.00, 'B3': 246.94,
        'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23, 'G4': 392.00, 'A4': 440.00, 'B4': 493.88,
        'C5': 523.25, 'D5': 587.33, 'E5': 659.25, 'F5': 698.46, 'G5': 783.99, 'A5': 880.00, 'B5': 987.77,
        'C6': 1046.50
    }
    
    # Creepy minor chord progressions for dungeon atmosphere
    CREEPY_CHORDS = {
        'minor_ii': ['D3', 'F3', 'A3'],
        'minor_v': ['G3', 'B3', 'D4'],
        'diminished': ['B2', 'D3', 'F3'],
        'augmented': ['C3', 'E3', 'G#3'],
        'wizard_chord': ['C2', 'F#2', 'B2', 'E3']  # Tritone for evil wizard feel
    }
    
    def __init__(self):
        self.available = NUMPY_AVAILABLE
        if not self.available:
            print("Warning: NumPy not available - cannot generate chiptune sounds")
    
    def generate_square_wave(self, frequency: float, duration: float, 
                           duty_cycle: float = 0.5, volume: float = 0.7) -> Optional[np.ndarray]:
        """
        Generate a square wave - the quintessential 8-bit sound
        
        Args:
            frequency: Frequency in Hz
            duration: Duration in seconds
            duty_cycle: Duty cycle (0.1-0.9, default 0.5 for classic square)
            volume: Volume level (0.0-1.0)
            
        Returns:
            NumPy array of audio samples or None if unavailable
        """
        if not self.available:
            return None
            
        t = np.linspace(0, duration, int(self.SAMPLE_RATE * duration), False)
        
        # Generate square wave using sine wave comparison
        wave = np.sign(np.sin(2 * np.pi * frequency * t + np.pi * (duty_cycle - 0.5)))
        
        # Apply volume and convert to 16-bit signed integers
        wave = (wave * volume * 32767).astype(np.int16)
        
        return wave
    
    def generate_triangle_wave(self, frequency: float, duration: float, 
                             volume: float = 0.7) -> Optional[np.ndarray]:
        """
        Generate a triangle wave for softer 8-bit sounds
        
        Args:
            frequency: Frequency in Hz
            duration: Duration in seconds
            volume: Volume level (0.0-1.0)
            
        Returns:
            NumPy array of audio samples or None if unavailable
        """
        if not self.available:
            return None
            
        t = np.linspace(0, duration, int(self.SAMPLE_RATE * duration), False)
        
        # Generate triangle wave using sawtooth and absolute value
        wave = 2 * np.abs(2 * (t * frequency - np.floor(t * frequency + 0.5))) - 1
        
        # Apply volume and convert to 16-bit signed integers
        wave = (wave * volume * 32767).astype(np.int16)
        
        return wave
    
    def generate_sawtooth_wave(self, frequency: float, duration: float, 
                              volume: float = 0.7) -> Optional[np.ndarray]:
        """
        Generate a sawtooth wave for harsh 8-bit sounds
        
        Args:
            frequency: Frequency in Hz
            duration: Duration in seconds
            volume: Volume level (0.0-1.0)
            
        Returns:
            NumPy array of audio samples or None if unavailable
        """
        if not self.available:
            return None
            
        t = np.linspace(0, duration, int(self.SAMPLE_RATE * duration), False)
        
        # Generate sawtooth wave
        wave = 2 * (t * frequency - np.floor(t * frequency + 0.5))
        
        # Apply volume and convert to 16-bit signed integers
        wave = (wave * volume * 32767).astype(np.int16)
        
        return wave
    
    def generate_noise(self, duration: float, volume: float = 0.3, 
                      filter_freq: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Generate noise for percussion and atmosphere
        
        Args:
            duration: Duration in seconds
            volume: Volume level (0.0-1.0)
            filter_freq: Optional low-pass filter frequency
            
        Returns:
            NumPy array of audio samples or None if unavailable
        """
        if not self.available:
            return None
            
        samples = int(self.SAMPLE_RATE * duration)
        
        # Generate white noise
        noise = np.random.uniform(-1, 1, samples)
        
        # Apply simple low-pass filter if requested
        if filter_freq:
            # Simple exponential moving average filter
            alpha = filter_freq / self.SAMPLE_RATE
            filtered = np.zeros_like(noise)
            filtered[0] = noise[0]
            for i in range(1, len(noise)):
                filtered[i] = alpha * noise[i] + (1 - alpha) * filtered[i-1]
            noise = filtered
        
        # Apply volume and convert to 16-bit signed integers
        noise = (noise * volume * 32767).astype(np.int16)
        
        return noise
    
    def apply_envelope(self, wave: np.ndarray, attack: float = 0.01, 
                      decay: float = 0.1, sustain: float = 0.7, 
                      release: float = 0.2) -> np.ndarray:
        """
        Apply ADSR envelope to a waveform
        
        Args:
            wave: Input waveform
            attack: Attack time in seconds
            decay: Decay time in seconds
            sustain: Sustain level (0.0-1.0)
            release: Release time in seconds
            
        Returns:
            Waveform with envelope applied
        """
        if not self.available:
            return wave
            
        length = len(wave)
        duration = length / self.SAMPLE_RATE
        
        # Calculate sample counts for each phase
        attack_samples = min(int(attack * self.SAMPLE_RATE), length // 4)
        decay_samples = min(int(decay * self.SAMPLE_RATE), length // 4)
        release_samples = min(int(release * self.SAMPLE_RATE), length // 4)
        sustain_samples = max(0, length - attack_samples - decay_samples - release_samples)
        
        envelope = np.ones(length)
        
        # Attack phase
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Decay phase
        if decay_samples > 0:
            start_idx = attack_samples
            end_idx = min(start_idx + decay_samples, length)
            actual_decay_samples = end_idx - start_idx
            if actual_decay_samples > 0:
                envelope[start_idx:end_idx] = np.linspace(1, sustain, actual_decay_samples)
        
        # Sustain phase
        if sustain_samples > 0:
            start_idx = attack_samples + decay_samples
            end_idx = min(start_idx + sustain_samples, length - release_samples)
            if end_idx > start_idx:
                envelope[start_idx:end_idx] = sustain
        
        # Release phase
        if release_samples > 0:
            start_idx = max(0, length - release_samples)
            actual_release_samples = length - start_idx
            if actual_release_samples > 0:
                envelope[start_idx:] = np.linspace(sustain, 0, actual_release_samples)
        
        return (wave * envelope).astype(np.int16)
    
    def generate_chord(self, notes: List[str], duration: float, 
                      waveform: str = 'square', volume: float = 0.5) -> Optional[np.ndarray]:
        """
        Generate a chord by mixing multiple notes
        
        Args:
            notes: List of note names (e.g., ['C3', 'E3', 'G3'])
            duration: Duration in seconds
            waveform: 'square', 'triangle', or 'sawtooth'
            volume: Volume level (0.0-1.0)
            
        Returns:
            Mixed waveform or None if unavailable
        """
        if not self.available or not notes:
            return None
        
        # Generate each note
        waves = []
        for note in notes:
            if note in self.NOTES:
                freq = self.NOTES[note]
                if waveform == 'square':
                    wave = self.generate_square_wave(freq, duration, volume=volume/len(notes))
                elif waveform == 'triangle':
                    wave = self.generate_triangle_wave(freq, duration, volume=volume/len(notes))
                elif waveform == 'sawtooth':
                    wave = self.generate_sawtooth_wave(freq, duration, volume=volume/len(notes))
                else:
                    continue
                    
                if wave is not None:
                    waves.append(wave)
        
        if not waves:
            return None
        
        # Mix all waves together
        mixed = np.sum(waves, axis=0)
        
        # Prevent clipping
        max_val = np.max(np.abs(mixed))
        if max_val > 32767:
            mixed = (mixed * 32767 / max_val).astype(np.int16)
        
        return mixed
    
    def create_menu_blip(self, pitch: str = 'C4', duration: float = 0.1) -> Optional[np.ndarray]:
        """Create a short menu navigation sound"""
        wave = self.generate_square_wave(self.NOTES[pitch], duration, duty_cycle=0.25, volume=0.6)
        if wave is not None:
            wave = self.apply_envelope(wave, attack=0.01, decay=0.05, sustain=0.3, release=0.04)
        return wave
    
    def create_menu_select(self) -> Optional[np.ndarray]:
        """Create a menu selection sound - ascending notes"""
        if not self.available:
            return None
            
        notes = ['C4', 'E4', 'G4']
        waves = []
        
        for i, note in enumerate(notes):
            wave = self.generate_square_wave(self.NOTES[note], 0.08, duty_cycle=0.5, volume=0.4)
            if wave is not None:
                wave = self.apply_envelope(wave, attack=0.005, decay=0.02, sustain=0.6, release=0.055)
                # Add slight delay between notes
                if i > 0:
                    silence = np.zeros(int(0.02 * self.SAMPLE_RATE), dtype=np.int16)
                    wave = np.concatenate([silence, wave])
                waves.append(wave)
        
        if waves:
            return np.concatenate(waves)
        return None
    
    def create_error_chord(self) -> Optional[np.ndarray]:
        """Create a dissonant error sound using tritone"""
        return self.generate_chord(['C3', 'F#3'], 0.3, waveform='square', volume=0.7)
    
    def create_wizard_spell(self) -> Optional[np.ndarray]:
        """Create a magical spell sound with frequency sweep"""
        if not self.available:
            return None
            
        duration = 0.8
        samples = int(duration * self.SAMPLE_RATE)
        t = np.linspace(0, duration, samples, False)
        
        # Frequency sweep from low to high (magical ascending effect)
        freq_start = 200
        freq_end = 800
        freq = freq_start + (freq_end - freq_start) * (t / duration) ** 2
        
        # Generate frequency-modulated square wave
        phase = np.cumsum(2 * np.pi * freq / self.SAMPLE_RATE)
        wave = np.sign(np.sin(phase))
        
        # Add some harmonic distortion for magical character
        wave += 0.3 * np.sign(np.sin(2 * phase))
        wave += 0.1 * np.sign(np.sin(3 * phase))
        
        # Apply envelope and volume
        wave = (wave * 0.4 * 32767).astype(np.int16)
        wave = self.apply_envelope(wave, attack=0.1, decay=0.2, sustain=0.6, release=0.5)
        
        return wave
    
    def create_dungeon_ambience(self, duration: float = 10.0) -> Optional[np.ndarray]:
        """Create looping dungeon ambience with low-frequency drones"""
        if not self.available:
            return None
            
        # Base drone at very low frequency
        drone1 = self.generate_square_wave(55, duration, duty_cycle=0.8, volume=0.2)  # Low A
        drone2 = self.generate_triangle_wave(73.42, duration, volume=0.15)  # Low D
        
        # Add some filtered noise for atmospheric texture
        noise = self.generate_noise(duration, volume=0.1, filter_freq=200)
        
        # Occasional low-frequency rumbles
        rumble_points = np.random.choice(int(duration * self.SAMPLE_RATE), 
                                       size=int(duration * 0.3), replace=False)
        rumble_wave = np.zeros(int(duration * self.SAMPLE_RATE), dtype=np.int16)
        
        for point in rumble_points:
            rumble_duration = 0.5
            rumble_samples = int(rumble_duration * self.SAMPLE_RATE)
            if point + rumble_samples < len(rumble_wave):
                rumble = self.generate_square_wave(36.71, rumble_duration, volume=0.1)  # Very low D
                if rumble is not None:
                    rumble_wave[point:point+rumble_samples] += rumble
        
        # Mix all components
        components = [drone1, drone2, noise, rumble_wave]
        components = [c for c in components if c is not None]
        
        if components:
            mixed = np.sum(components, axis=0)
            # Normalize to prevent clipping
            max_val = np.max(np.abs(mixed))
            if max_val > 32767:
                mixed = (mixed * 32767 / max_val).astype(np.int16)
            return mixed
        
        return None
    
    def save_to_wav(self, wave: np.ndarray, filename: str, 
                   output_dir: Optional[str] = None) -> bool:
        """
        Save waveform to WAV file
        
        Args:
            wave: Audio waveform to save
            filename: Output filename (without extension)
            output_dir: Output directory (defaults to audio/sounds/effects)
            
        Returns:
            True if saved successfully
        """
        if not PYGAME_AVAILABLE or wave is None:
            return False
        
        try:
            # Initialize pygame mixer if not already done
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=self.SAMPLE_RATE, size=-16, channels=2, buffer=512)
            
            if output_dir is None:
                output_dir = Path(__file__).parent / "sounds" / "effects"
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{filename}.wav"
            
            # Convert mono to stereo for pygame compatibility
            if len(wave.shape) == 1:
                stereo_wave = np.column_stack((wave, wave))
            else:
                stereo_wave = wave
            
            # Try to use scipy for WAV export first
            try:
                from scipy.io import wavfile
                wavfile.write(str(output_path), self.SAMPLE_RATE, stereo_wave)
                return True
            except ImportError:
                # Fallback: Create pygame sound and use a simple WAV writer
                try:
                    import struct
                    import wave as wav_module
                    
                    with wav_module.open(str(output_path), 'wb') as wav_file:
                        wav_file.setnchannels(2)
                        wav_file.setsampwidth(2)  # 16-bit
                        wav_file.setframerate(self.SAMPLE_RATE)
                        
                        # Convert to bytes
                        audio_data = []
                        for frame in stereo_wave:
                            audio_data.append(struct.pack('<h', frame[0]))  # Left channel
                            audio_data.append(struct.pack('<h', frame[1]))  # Right channel
                        
                        wav_file.writeframes(b''.join(audio_data))
                    
                    return True
                except Exception as fallback_error:
                    print(f"Warning: Cannot save {filename}.wav - {fallback_error}")
                    return False
                
        except Exception as e:
            print(f"Error saving {filename}.wav: {e}")
            return False

# Convenience functions for common sound effects
def generate_all_basic_sounds(generator: ChiptuneGenerator, output_dir: str) -> List[str]:
    """Generate all basic sound effects and save them"""
    generated = []
    
    # Menu sounds
    sounds = {
        'menu_blip': generator.create_menu_blip(),
        'menu_select': generator.create_menu_select(),
        'error_chord': generator.create_error_chord(),
        'wizard_spell': generator.create_wizard_spell(),
    }
    
    for name, wave in sounds.items():
        if wave is not None and generator.save_to_wav(wave, name, output_dir):
            generated.append(name)
    
    # Generate ambient track
    ambient_dir = Path(output_dir).parent / "ambient"
    ambient = generator.create_dungeon_ambience(30.0)  # 30-second loop
    if ambient is not None and generator.save_to_wav(ambient, "dungeon_base", str(ambient_dir)):
        generated.append("dungeon_base")
    
    return generated

if __name__ == "__main__":
    """Generate basic sound library when run directly"""
    generator = ChiptuneGenerator()
    
    if not generator.available:
        print("Error: NumPy required for sound generation")
        sys.exit(1)
    
    output_dir = Path(__file__).parent / "sounds" / "effects"
    generated = generate_all_basic_sounds(generator, str(output_dir))
    
    print(f"Generated {len(generated)} sound effects:")
    for sound in generated:
        print(f"  - {sound}")