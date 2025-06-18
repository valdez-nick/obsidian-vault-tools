#!/usr/bin/env python3
"""
Sound Library for Enhanced Vault Manager
Manages and generates the complete sound effect library for creepy dungeon crawler experience
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import random

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

from chiptune_generator import ChiptuneGenerator

class SoundLibrary:
    """
    Complete sound library for the Enhanced Vault Manager
    Provides all sound effects for creepy dungeon crawler atmosphere
    """
    
    def __init__(self):
        self.generator = ChiptuneGenerator()
        self.sounds_dir = Path(__file__).parent / "sounds"
        self.effects_dir = self.sounds_dir / "effects"
        self.ambient_dir = self.sounds_dir / "ambient"
        self.voices_dir = self.sounds_dir / "voices"
        
        # Ensure directories exist
        for directory in [self.effects_dir, self.ambient_dir, self.voices_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def generate_menu_sounds(self) -> List[str]:
        """Generate all menu navigation sound effects"""
        generated = []
        
        if not self.generator.available:
            return generated
        
        # Basic menu blip - short and crisp
        menu_blip = self.generator.create_menu_blip('C5', 0.08)
        if menu_blip is not None:
            if self.generator.save_to_wav(menu_blip, 'menu_blip', str(self.effects_dir)):
                generated.append('menu_blip')
        
        # Menu selection - ascending arpeggio
        menu_select = self.generator.create_menu_select()
        if menu_select is not None:
            if self.generator.save_to_wav(menu_select, 'menu_select', str(self.effects_dir)):
                generated.append('menu_select')
        
        # Menu back - descending notes
        back_notes = ['G4', 'E4', 'C4']
        waves = []
        for note in back_notes:
            wave = self.generator.generate_square_wave(
                self.generator.NOTES[note], 0.06, duty_cycle=0.3, volume=0.4
            )
            if wave is not None:
                wave = self.generator.apply_envelope(wave, attack=0.005, decay=0.02, sustain=0.5, release=0.035)
                waves.append(wave)
        
        if waves:
            menu_back = np.concatenate(waves)
            if self.generator.save_to_wav(menu_back, 'menu_back', str(self.effects_dir)):
                generated.append('menu_back')
        
        # Error sound - dissonant chord
        error_chord = self.generator.create_error_chord()
        if error_chord is not None:
            if self.generator.save_to_wav(error_chord, 'error_chord', str(self.effects_dir)):
                generated.append('error_chord')
        
        return generated
    
    def generate_vault_operation_sounds(self) -> List[str]:
        """Generate sounds for vault operations (scan, backup, etc.)"""
        generated = []
        
        if not self.generator.available:
            return generated
        
        # Scan begin - rising frequency sweep
        scan_begin = self._create_frequency_sweep(150, 400, 0.5, 'square')
        if scan_begin is not None:
            if self.generator.save_to_wav(scan_begin, 'scan_begin', str(self.effects_dir)):
                generated.append('scan_begin')
        
        # Scan tick - short pulse for progress
        scan_tick = self.generator.generate_square_wave(800, 0.03, duty_cycle=0.1, volume=0.3)
        if scan_tick is not None:
            scan_tick = self.generator.apply_envelope(scan_tick, attack=0.001, decay=0.01, sustain=0.2, release=0.019)
            if self.generator.save_to_wav(scan_tick, 'scan_tick', str(self.effects_dir)):
                generated.append('scan_tick')
        
        # Scan complete - success chord
        success_chord = self.generator.generate_chord(['C4', 'E4', 'G4', 'C5'], 0.6, 'triangle', 0.6)
        if success_chord is not None:
            if self.generator.save_to_wav(success_chord, 'scan_complete', str(self.effects_dir)):
                generated.append('scan_complete')
        
        # Backup sounds
        backup_begin = self._create_digital_noise_burst(0.3, 1200, 0.4)
        if backup_begin is not None:
            if self.generator.save_to_wav(backup_begin, 'backup_begin', str(self.effects_dir)):
                generated.append('backup_begin')
        
        backup_complete = self.generator.generate_chord(['F4', 'A4', 'C5'], 0.8, 'square', 0.5)
        if backup_complete is not None:
            if self.generator.save_to_wav(backup_complete, 'backup_complete', str(self.effects_dir)):
                generated.append('backup_complete')
        
        return generated
    
    def generate_magical_sounds(self) -> List[str]:
        """Generate wizard and magical sound effects"""
        generated = []
        
        if not self.generator.available:
            return generated
        
        # Magic begin - spell casting start
        magic_begin = self.generator.create_wizard_spell()
        if magic_begin is not None:
            if self.generator.save_to_wav(magic_begin, 'magic_begin', str(self.effects_dir)):
                generated.append('magic_begin')
        
        # Magic complete - successful spell
        magic_complete = self._create_shimmer_effect(1.0)
        if magic_complete is not None:
            if self.generator.save_to_wav(magic_complete, 'magic_complete', str(self.effects_dir)):
                generated.append('magic_complete')
        
        # Magic reveal - discovery sound
        magic_reveal = self._create_bell_tone('C6', 0.8)
        if magic_reveal is not None:
            if self.generator.save_to_wav(magic_reveal, 'magic_reveal', str(self.effects_dir)):
                generated.append('magic_reveal')
        
        # Wizard greeting - friendly magical sound
        wizard_hello = self._create_magical_arpeggio(['C4', 'E4', 'G4', 'B4', 'D5'], 0.4)
        if wizard_hello is not None:
            if self.generator.save_to_wav(wizard_hello, 'wizard_hello', str(self.effects_dir)):
                generated.append('wizard_hello')
        
        # Wizard warning - ominous tone
        wizard_warn = self.generator.generate_chord(['C3', 'F#3', 'Bb3'], 0.6, 'sawtooth', 0.6)
        if wizard_warn is not None:
            if self.generator.save_to_wav(wizard_warn, 'wizard_warn', str(self.effects_dir)):
                generated.append('wizard_warn')
        
        # Spell cast - quick magical burst
        spell_cast = self._create_frequency_sweep(400, 1200, 0.2, 'triangle')
        if spell_cast is not None:
            if self.generator.save_to_wav(spell_cast, 'spell_cast', str(self.effects_dir)):
                generated.append('spell_cast')
        
        # Magic success - triumphant chord
        magic_success = self.generator.generate_chord(['C4', 'E4', 'G4', 'C5'], 1.0, 'square', 0.7)
        if magic_success is not None:
            if self.generator.save_to_wav(magic_success, 'magic_success', str(self.effects_dir)):
                generated.append('magic_success')
        
        return generated
    
    def generate_dungeon_sounds(self) -> List[str]:
        """Generate creepy dungeon atmosphere sound effects"""
        generated = []
        
        if not self.generator.available:
            return generated
        
        # Door creak - slow frequency bend
        door_creak = self._create_creaking_sound(1.5)
        if door_creak is not None:
            if self.generator.save_to_wav(door_creak, 'door_creak', str(self.effects_dir)):
                generated.append('door_creak')
        
        # Stone step - percussive thunk
        stone_step = self._create_percussion_hit(80, 0.2)
        if stone_step is not None:
            if self.generator.save_to_wav(stone_step, 'stone_step', str(self.effects_dir)):
                generated.append('stone_step')
        
        # Danger sting - sharp warning
        danger_notes = ['C3', 'F#3']  # Tritone for maximum tension
        danger_sting = self.generator.generate_chord(danger_notes, 0.4, 'sawtooth', 0.8)
        if danger_sting is not None:
            if self.generator.save_to_wav(danger_sting, 'danger_sting', str(self.effects_dir)):
                generated.append('danger_sting')
        
        # Item found - discovery chime
        item_found = self._create_bell_sequence(['G5', 'C6', 'E6'], 0.15)
        if item_found is not None:
            if self.generator.save_to_wav(item_found, 'item_found', str(self.effects_dir)):
                generated.append('item_found')
        
        return generated
    
    def generate_ambient_tracks(self) -> List[str]:
        """Generate looping ambient background tracks"""
        generated = []
        
        if not self.generator.available:
            return generated
        
        # Main dungeon ambience
        dungeon_base = self.generator.create_dungeon_ambience(30.0)
        if dungeon_base is not None:
            if self.generator.save_to_wav(dungeon_base, 'dungeon_base', str(self.ambient_dir)):
                generated.append('dungeon_base')
        
        # Deeper dungeon level - more ominous
        deep_dungeon = self._create_deep_ambience(45.0)
        if deep_dungeon is not None:
            if self.generator.save_to_wav(deep_dungeon, 'deep_dungeon', str(self.ambient_dir)):
                generated.append('deep_dungeon')
        
        # Wizard tower - magical atmosphere
        wizard_tower = self._create_magical_ambience(40.0)
        if wizard_tower is not None:
            if self.generator.save_to_wav(wizard_tower, 'wizard_tower', str(self.ambient_dir)):
                generated.append('wizard_tower')
        
        return generated
    
    def _create_frequency_sweep(self, start_freq: float, end_freq: float, 
                               duration: float, waveform: str = 'square') -> Optional[np.ndarray]:
        """Create a frequency sweep effect"""
        if not self.generator.available:
            return None
            
        samples = int(duration * self.generator.SAMPLE_RATE)
        t = np.linspace(0, duration, samples, False)
        
        # Logarithmic frequency sweep for more natural sound
        freq = start_freq * (end_freq / start_freq) ** (t / duration)
        
        # Generate waveform
        phase = np.cumsum(2 * np.pi * freq / self.generator.SAMPLE_RATE)
        
        if waveform == 'square':
            wave = np.sign(np.sin(phase))
        elif waveform == 'triangle':
            wave = 2 * np.abs(2 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5))) - 1
        elif waveform == 'sawtooth':
            wave = 2 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5))
        else:
            wave = np.sin(phase)
        
        # Apply envelope and volume
        wave = (wave * 0.5 * 32767).astype(np.int16)
        wave = self.generator.apply_envelope(wave, attack=0.05, decay=0.1, sustain=0.7, release=0.15)
        
        return wave
    
    def _create_digital_noise_burst(self, duration: float, cutoff_freq: float, volume: float) -> Optional[np.ndarray]:
        """Create filtered noise burst for digital/computer sounds"""
        noise = self.generator.generate_noise(duration, volume, filter_freq=cutoff_freq)
        if noise is not None:
            noise = self.generator.apply_envelope(noise, attack=0.01, decay=0.1, sustain=0.4, release=0.45)
        return noise
    
    def _create_shimmer_effect(self, duration: float) -> Optional[np.ndarray]:
        """Create a shimmering magical effect with multiple frequencies"""
        if not self.generator.available:
            return None
            
        # Multiple high-frequency oscillators with slight detuning
        freqs = [1200, 1205, 1198, 1600, 1602]
        waves = []
        
        for freq in freqs:
            wave = self.generator.generate_triangle_wave(freq, duration, volume=0.2)
            if wave is not None:
                # Apply tremolo effect
                t = np.linspace(0, duration, len(wave), False)
                tremolo = 0.8 + 0.2 * np.sin(2 * np.pi * 5 * t)  # 5Hz tremolo
                wave = (wave * tremolo).astype(np.int16)
                waves.append(wave)
        
        if waves:
            mixed = np.sum(waves, axis=0)
            # Apply sparkle envelope
            mixed = self.generator.apply_envelope(mixed, attack=0.1, decay=0.3, sustain=0.4, release=0.6)
            return mixed
        
        return None
    
    def _create_bell_tone(self, note: str, duration: float) -> Optional[np.ndarray]:
        """Create bell-like tone with harmonics"""
        if not self.generator.available or note not in self.generator.NOTES:
            return None
            
        fundamental = self.generator.NOTES[note]
        
        # Bell harmonics (fundamental + overtones)
        harmonics = [
            (fundamental, 1.0),
            (fundamental * 2.76, 0.6),
            (fundamental * 5.4, 0.25),
            (fundamental * 8.93, 0.1)
        ]
        
        waves = []
        for freq, amplitude in harmonics:
            wave = self.generator.generate_triangle_wave(freq, duration, volume=amplitude * 0.3)
            if wave is not None:
                waves.append(wave)
        
        if waves:
            mixed = np.sum(waves, axis=0)
            # Bell-like envelope with long decay
            mixed = self.generator.apply_envelope(mixed, attack=0.01, decay=0.2, sustain=0.3, release=0.8)
            return mixed
        
        return None
    
    def _create_magical_arpeggio(self, notes: List[str], note_duration: float) -> Optional[np.ndarray]:
        """Create ascending magical arpeggio"""
        waves = []
        
        for i, note in enumerate(notes):
            if note in self.generator.NOTES:
                wave = self.generator.generate_triangle_wave(
                    self.generator.NOTES[note], note_duration, volume=0.5
                )
                if wave is not None:
                    wave = self.generator.apply_envelope(wave, attack=0.01, decay=0.05, sustain=0.7, release=0.24)
                    # Add slight delay between notes
                    if i > 0:
                        delay = np.zeros(int(0.05 * self.generator.SAMPLE_RATE), dtype=np.int16)
                        wave = np.concatenate([delay, wave])
                    waves.append(wave)
        
        if waves:
            return np.concatenate(waves)
        return None
    
    def _create_bell_sequence(self, notes: List[str], note_duration: float) -> Optional[np.ndarray]:
        """Create sequence of bell tones"""
        waves = []
        
        for note in notes:
            bell = self._create_bell_tone(note, note_duration)
            if bell is not None:
                waves.append(bell)
                # Small gap between bells
                silence = np.zeros(int(0.02 * self.generator.SAMPLE_RATE), dtype=np.int16)
                waves.append(silence)
        
        if waves:
            return np.concatenate(waves)
        return None
    
    def _create_creaking_sound(self, duration: float) -> Optional[np.ndarray]:
        """Create creaking door sound with frequency bends"""
        if not self.generator.available:
            return None
            
        # Slow frequency bend with some randomness
        samples = int(duration * self.generator.SAMPLE_RATE)
        t = np.linspace(0, duration, samples, False)
        
        # Base frequency with random variations
        base_freq = 150
        freq_variation = 30 * np.sin(2 * np.pi * 0.8 * t) + 10 * np.random.random(samples)
        freq = base_freq + freq_variation
        
        # Generate creaky sawtooth wave
        phase = np.cumsum(2 * np.pi * freq / self.generator.SAMPLE_RATE)
        wave = 2 * (phase / (2 * np.pi) - np.floor(phase / (2 * np.pi) + 0.5))
        
        # Add some noise for texture
        noise = np.random.uniform(-0.2, 0.2, samples)
        wave = wave + noise
        
        # Apply volume and envelope
        wave = (wave * 0.4 * 32767).astype(np.int16)
        wave = self.generator.apply_envelope(wave, attack=0.2, decay=0.3, sustain=0.6, release=0.5)
        
        return wave
    
    def _create_percussion_hit(self, frequency: float, duration: float) -> Optional[np.ndarray]:
        """Create percussive hit sound"""
        if not self.generator.available:
            return None
            
        # Short burst of low frequency with noise
        tone = self.generator.generate_square_wave(frequency, duration, duty_cycle=0.8, volume=0.6)
        noise = self.generator.generate_noise(duration, volume=0.3, filter_freq=500)
        
        if tone is not None and noise is not None:
            mixed = (tone + noise) // 2
            # Sharp attack, quick decay
            mixed = self.generator.apply_envelope(mixed, attack=0.001, decay=0.05, sustain=0.1, release=0.149)
            return mixed
        
        return tone or noise
    
    def _create_deep_ambience(self, duration: float) -> Optional[np.ndarray]:
        """Create deeper, more ominous ambient track"""
        if not self.generator.available:
            return None
            
        # Very low frequency drones
        drone1 = self.generator.generate_square_wave(36.71, duration, duty_cycle=0.9, volume=0.15)  # Very low D
        drone2 = self.generator.generate_triangle_wave(41.20, duration, volume=0.12)  # Very low E
        
        # Occasional distant rumbles
        rumbles = self._add_random_rumbles(duration, frequency=30, intensity=0.08)
        
        # Dark noise texture
        dark_noise = self.generator.generate_noise(duration, volume=0.06, filter_freq=150)
        
        components = [drone1, drone2, rumbles, dark_noise]
        components = [c for c in components if c is not None]
        
        if components:
            mixed = np.sum(components, axis=0)
            # Normalize
            max_val = np.max(np.abs(mixed))
            if max_val > 32767:
                mixed = (mixed * 32767 / max_val).astype(np.int16)
            return mixed
        
        return None
    
    def _create_magical_ambience(self, duration: float) -> Optional[np.ndarray]:
        """Create magical tower ambient atmosphere"""
        if not self.generator.available:
            return None
            
        # Higher frequency mystical drones
        drone1 = self.generator.generate_triangle_wave(220, duration, volume=0.1)  # A3
        drone2 = self.generator.generate_square_wave(329.63, duration, duty_cycle=0.3, volume=0.08)  # E4
        
        # Occasional magical sparkles
        sparkles = self._add_random_sparkles(duration, intensity=0.1)
        
        # Light magical noise
        magic_noise = self.generator.generate_noise(duration, volume=0.04, filter_freq=800)
        
        components = [drone1, drone2, sparkles, magic_noise]
        components = [c for c in components if c is not None]
        
        if components:
            mixed = np.sum(components, axis=0)
            # Normalize
            max_val = np.max(np.abs(mixed))
            if max_val > 32767:
                mixed = (mixed * 32767 / max_val).astype(np.int16)
            return mixed
        
        return None
    
    def _add_random_rumbles(self, duration: float, frequency: float, intensity: float) -> Optional[np.ndarray]:
        """Add random low rumbles to ambient track"""
        if not self.generator.available:
            return None
            
        samples = int(duration * self.generator.SAMPLE_RATE)
        rumble_track = np.zeros(samples, dtype=np.int16)
        
        # Add rumbles at random intervals
        rumble_count = int(duration * 0.2)  # About every 5 seconds
        rumble_points = np.random.choice(samples - int(1.0 * self.generator.SAMPLE_RATE), 
                                       size=rumble_count, replace=False)
        
        for point in rumble_points:
            rumble_duration = random.uniform(0.5, 2.0)
            rumble_samples = int(rumble_duration * self.generator.SAMPLE_RATE)
            
            if point + rumble_samples < samples:
                rumble = self.generator.generate_square_wave(frequency, rumble_duration, volume=intensity)
                if rumble is not None:
                    rumble = self.generator.apply_envelope(rumble, attack=0.2, decay=0.3, sustain=0.4, release=0.5)
                    rumble_track[point:point+rumble_samples] += rumble
        
        return rumble_track
    
    def _add_random_sparkles(self, duration: float, intensity: float) -> Optional[np.ndarray]:
        """Add random high-frequency sparkles to magical ambient"""
        if not self.generator.available:
            return None
            
        samples = int(duration * self.generator.SAMPLE_RATE)
        sparkle_track = np.zeros(samples, dtype=np.int16)
        
        # Add sparkles at random intervals
        sparkle_count = int(duration * 0.5)  # About every 2 seconds
        sparkle_points = np.random.choice(samples - int(0.5 * self.generator.SAMPLE_RATE), 
                                        size=sparkle_count, replace=False)
        
        high_freqs = [1200, 1600, 2000, 2400]
        
        for point in sparkle_points:
            freq = random.choice(high_freqs)
            sparkle_duration = random.uniform(0.1, 0.4)
            sparkle_samples = int(sparkle_duration * self.generator.SAMPLE_RATE)
            
            if point + sparkle_samples < samples:
                sparkle = self.generator.generate_triangle_wave(freq, sparkle_duration, volume=intensity)
                if sparkle is not None:
                    sparkle = self.generator.apply_envelope(sparkle, attack=0.02, decay=0.1, sustain=0.3, release=0.5)
                    sparkle_track[point:point+sparkle_samples] += sparkle
        
        return sparkle_track
    
    def generate_all_sounds(self) -> Dict[str, List[str]]:
        """Generate the complete sound library"""
        results = {
            'menu': self.generate_menu_sounds(),
            'vault_ops': self.generate_vault_operation_sounds(),
            'magical': self.generate_magical_sounds(),
            'dungeon': self.generate_dungeon_sounds(),
            'ambient': self.generate_ambient_tracks()
        }
        
        total_generated = sum(len(sounds) for sounds in results.values())
        print(f"\nGenerated {total_generated} total sound effects:")
        
        for category, sounds in results.items():
            if sounds:
                print(f"\n{category.title()} sounds ({len(sounds)}):")
                for sound in sounds:
                    print(f"  ✓ {sound}")
        
        return results

if __name__ == "__main__":
    """Generate complete sound library when run directly"""
    library = SoundLibrary()
    
    if not library.generator.available:
        print("Error: NumPy required for sound generation")
        sys.exit(1)
    
    print("🧙‍♂️ Generating Dungeon Crawler Sound Library...")
    print("=" * 50)
    
    results = library.generate_all_sounds()
    
    print("\n" + "=" * 50)
    print("🎵 Sound library generation complete!")
    print("Ready for creepy dungeon crawler atmosphere! 🏰")