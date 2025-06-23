#!/usr/bin/env python3
"""
Audio Manager for Enhanced Vault Manager
Provides 8-bit retro dungeon crawler atmosphere with wizards and creepy ambiance
"""

import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Any, List
import random
from obsidian_vault_tools.memory import track_tool_usage, memory_cached

# Try to import pygame for audio
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# Try to import numpy for sound generation
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

class AudioManager:
    """
    Central audio management system for the Enhanced Vault Manager
    Provides atmospheric 8-bit dungeon crawler sounds with wizard themes
    """
    
    def __init__(self, enable_audio: bool = True):
        self.enabled = enable_audio and PYGAME_AVAILABLE
        self.initialized = False
        self.sound_cache: Dict[str, Any] = {}
        self.background_thread: Optional[threading.Thread] = None
        self.ambient_playing = False
        self.volumes = {
            'master': 0.7,
            'effects': 0.8,
            'ambient': 0.3,
            'voice': 0.6
        }
        
        if self.enabled:
            self._initialize_pygame()
    
    def _initialize_pygame(self) -> bool:
        """Initialize pygame mixer for audio playback"""
        try:
            pygame.mixer.pre_init(
                frequency=22050,    # Lower frequency for retro feel
                size=-16,           # 16-bit signed samples
                channels=2,         # Stereo
                buffer=512          # Small buffer for responsiveness
            )
            pygame.mixer.init()
            pygame.mixer.set_num_channels(8)  # Classic 8-channel mixing
            self.initialized = True
            return True
        except Exception as e:
            print(f"Audio initialization failed: {e}")
            self.enabled = False
            return False
    
    @track_tool_usage(category="audio")
    def play_effect(self, effect_name: str, volume_override: Optional[float] = None, stop_previous: bool = True) -> bool:
        """
        Play a sound effect
        
        Args:
            effect_name: Name of the effect to play
            volume_override: Optional volume override (0.0 to 1.0)
            stop_previous: Whether to stop currently playing effects
            
        Returns:
            True if sound played successfully
        """
        if not self.enabled:
            return False
            
        try:
            # Stop any currently playing effects if requested
            if stop_previous:
                # Stop all channels except channel 0 (reserved for ambient)
                for channel_id in range(1, pygame.mixer.get_num_channels()):
                    channel = pygame.mixer.Channel(channel_id)
                    if channel.get_busy():
                        channel.stop()
            
            sound = self._get_sound(effect_name)
            if sound:
                volume = volume_override or (self.volumes['master'] * self.volumes['effects'])
                sound.set_volume(volume)
                
                # Try to find a free channel (not channel 0)
                channel = None
                for channel_id in range(1, pygame.mixer.get_num_channels()):
                    test_channel = pygame.mixer.Channel(channel_id)
                    if not test_channel.get_busy():
                        channel = test_channel
                        break
                
                if channel:
                    channel.play(sound)
                else:
                    # If no free channel, use any available channel
                    sound.play()
                
                return True
        except Exception as e:
            print(f"Error playing effect {effect_name}: {e}")
        
        return False
    
    @track_tool_usage(category="audio")
    def start_ambient(self, ambient_name: str = "dungeon_base") -> bool:
        """
        Start ambient background music/atmosphere
        
        Args:
            ambient_name: Name of ambient track to play
            
        Returns:
            True if ambient started successfully
        """
        if not self.enabled or self.ambient_playing:
            return False
            
        try:
            self.ambient_playing = True
            self.background_thread = threading.Thread(
                target=self._ambient_loop,
                args=(ambient_name,),
                daemon=True
            )
            self.background_thread.start()
            return True
        except Exception as e:
            print(f"Error starting ambient {ambient_name}: {e}")
            return False
    
    def stop_ambient(self) -> None:
        """Stop ambient background audio"""
        self.ambient_playing = False
        if hasattr(pygame.mixer, 'music'):
            pygame.mixer.music.stop()
    
    def _ambient_loop(self, ambient_name: str) -> None:
        """Background thread for ambient audio playback"""
        try:
            ambient_path = Path(__file__).parent / "sounds" / "ambient" / f"{ambient_name}.wav"
            if ambient_path.exists():
                # Use the music channel for ambient (separate from effects)
                pygame.mixer.music.load(str(ambient_path))
                volume = self.volumes['master'] * self.volumes['ambient']
                pygame.mixer.music.set_volume(volume)
                pygame.mixer.music.play(-1)  # Loop indefinitely
            else:
                # Generate procedural ambient if file doesn't exist
                self._generate_ambient_loop(ambient_name)
        except Exception as e:
            print(f"Error in ambient loop: {e}")
    
    def _generate_ambient_loop(self, ambient_name: str) -> None:
        """Generate procedural ambient sounds when audio files don't exist"""
        if not NUMPY_AVAILABLE:
            return
            
        # This will be implemented with the chiptune generator
        # For now, just play silence
        pass
    
    @memory_cached(ttl=3600)  # Cache sounds for 1 hour
    def _get_sound(self, sound_name: str) -> Optional[Any]:
        """
        Get sound from cache or load it
        
        Args:
            sound_name: Name of the sound to load
            
        Returns:
            pygame.mixer.Sound object or None
        """
        if sound_name in self.sound_cache:
            return self.sound_cache[sound_name]
        
        # Try to load from effects directory
        sound_path = Path(__file__).parent / "sounds" / "effects" / f"{sound_name}.wav"
        
        if sound_path.exists():
            try:
                sound = pygame.mixer.Sound(str(sound_path))
                self.sound_cache[sound_name] = sound
                return sound
            except Exception as e:
                print(f"Error loading sound {sound_name}: {e}")
        
        # If sound file doesn't exist, try to generate it procedurally
        return self._generate_sound(sound_name)
    
    def _generate_sound(self, sound_name: str) -> Optional[Any]:
        """
        Generate sound effects procedurally if not found
        Will use the chiptune generator when implemented
        """
        # This will be implemented with the chiptune generator
        return None
    
    def set_volume(self, category: str, volume: float) -> None:
        """
        Set volume for a category
        
        Args:
            category: 'master', 'effects', 'ambient', or 'voice'
            volume: Volume level (0.0 to 1.0)
        """
        if category in self.volumes:
            self.volumes[category] = max(0.0, min(1.0, volume))
    
    def get_volume(self, category: str) -> float:
        """Get volume for a category"""
        return self.volumes.get(category, 0.0)
    
    def is_available(self) -> bool:
        """Check if audio system is available and initialized"""
        return self.enabled and self.initialized
    
    def cleanup(self) -> None:
        """Clean up audio resources"""
        self.stop_ambient()
        if self.initialized:
            pygame.mixer.quit()

# Predefined sound effect mappings for different UI actions
SOUND_EFFECTS = {
    # Menu navigation
    'menu_move': 'menu_blip',
    'menu_select': 'menu_select',
    'menu_back': 'menu_back',
    'menu_error': 'error_chord',
    
    # Vault operations
    'vault_scan_start': 'scan_begin',
    'vault_scan_progress': 'scan_tick',
    'vault_scan_complete': 'scan_complete',
    'backup_start': 'backup_begin',
    'backup_complete': 'backup_complete',
    
    # ASCII art operations
    'ascii_generate_start': 'magic_begin',
    'ascii_generate_complete': 'magic_complete',
    'ascii_reveal': 'magic_reveal',
    
    # Wizard/magical sounds
    'wizard_greeting': 'wizard_hello',
    'wizard_warning': 'wizard_warn',
    'wizard_spell': 'spell_cast',
    'magical_success': 'magic_success',
    
    # Creepy dungeon atmosphere
    'dungeon_door': 'door_creak',
    'dungeon_step': 'stone_step',
    'dungeon_danger': 'danger_sting',
    'dungeon_discovery': 'item_found'
}

# Global audio manager instance
_audio_manager: Optional[AudioManager] = None

def get_audio_manager() -> AudioManager:
    """Get the global audio manager instance"""
    global _audio_manager
    if _audio_manager is None:
        _audio_manager = AudioManager()
    return _audio_manager

def play_sound(effect_name: str, volume: Optional[float] = None) -> bool:
    """Convenience function to play a sound effect"""
    return get_audio_manager().play_effect(effect_name, volume)

def start_dungeon_ambiance() -> bool:
    """Start the creepy dungeon atmosphere"""
    return get_audio_manager().start_ambient("dungeon_base")

def stop_dungeon_ambiance() -> None:
    """Stop the dungeon atmosphere"""
    get_audio_manager().stop_ambient()

def wizard_greeting() -> bool:
    """Play wizard greeting sound"""
    return play_sound('wizard_greeting')

def magical_success() -> bool:
    """Play magical success sound"""
    return play_sound('magical_success')

def dungeon_danger() -> bool:
    """Play danger sting for warnings"""
    return play_sound('dungeon_danger')