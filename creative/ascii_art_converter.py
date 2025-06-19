#!/usr/bin/env python3
"""
Advanced ASCII Art Converter
Converts images to detailed ASCII art using various techniques
"""

import os
import sys
from pathlib import Path
from typing import Tuple, List

# Check for required packages
try:
    from PIL import Image, ImageEnhance, ImageFilter
    import numpy as np
except ImportError:
    print("Installing required packages...")
    os.system(f"{sys.executable} -m pip install pillow numpy")
    from PIL import Image, ImageEnhance, ImageFilter
    import numpy as np

class ASCIIArtConverter:
    """Convert images to detailed ASCII art with multiple style options"""
    
    # Different character sets for various styles
    CHAR_SETS = {
        'standard': ' .:-=+*#%@',
        'detailed': ' .,:-;!~=+*#%$@',
        'blocks': ' ░▒▓█',
        'extended': ' `^",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$',
        'simple': ' .oO@',
        'dore': ' \'`^",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$'
    }
    
    def __init__(self, char_set='detailed', width=80, enhance_contrast=True):
        """
        Initialize the converter
        
        Args:
            char_set: Character set to use ('standard', 'detailed', 'blocks', 'extended', 'dore')
            width: Target width in characters
            enhance_contrast: Whether to enhance image contrast
        """
        self.chars = self.CHAR_SETS.get(char_set, self.CHAR_SETS['detailed'])
        self.width = width
        self.enhance_contrast = enhance_contrast
    
    def image_to_ascii(self, image_path: str, output_path: str = None) -> str:
        """
        Convert an image to ASCII art
        
        Args:
            image_path: Path to input image
            output_path: Optional path to save ASCII art
            
        Returns:
            ASCII art as string
        """
        # Load and prepare image
        img = Image.open(image_path)
        
        # Convert to grayscale
        img = img.convert('L')
        
        # Enhance contrast if requested
        if self.enhance_contrast:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(2.0)
        
        # Calculate height to maintain aspect ratio
        aspect_ratio = img.height / img.width
        height = int(aspect_ratio * self.width * 0.55)  # 0.55 compensates for character height
        
        # Resize image
        img = img.resize((self.width, height), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        pixels = np.array(img)
        
        # Map pixels to characters
        ascii_art = self._pixels_to_ascii(pixels)
        
        # Save if output path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(ascii_art)
        
        return ascii_art
    
    def _pixels_to_ascii(self, pixels: np.ndarray) -> str:
        """Convert pixel array to ASCII characters"""
        ascii_art = []
        
        for row in pixels:
            ascii_row = ''
            for pixel in row:
                # Map pixel value (0-255) to character
                char_index = int(pixel * (len(self.chars) - 1) / 255)
                ascii_row += self.chars[char_index]
            ascii_art.append(ascii_row)
        
        return '\n'.join(ascii_art)
    
    def create_dore_style_ascii(self, image_path: str) -> str:
        """
        Create ASCII art in the style of Gustave Doré engravings
        Uses edge detection and cross-hatching patterns
        """
        img = Image.open(image_path)
        img = img.convert('L')
        
        # Apply edge detection
        edges = img.filter(ImageFilter.FIND_EDGES)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(edges)
        edges = enhancer.enhance(3.0)
        
        # Resize
        aspect_ratio = img.height / img.width
        height = int(aspect_ratio * self.width * 0.55)
        edges = edges.resize((self.width, height), Image.Resampling.LANCZOS)
        
        # Convert to array
        pixels = np.array(edges)
        
        # Use special character set for Doré style
        self.chars = self.CHAR_SETS['dore']
        
        return self._pixels_to_ascii(pixels)

def demo_satan_ascii():
    """Demo: Create detailed Satan/Lucifer ASCII art"""
    
    # Using extended character set for maximum detail
    satan_art = r"""
                                    ___,,,___
                              _,-='=- =-  -`"--.__,,.._
                           ,-;// /  - -       -   -= - "=.
                         ,'///    -     -   -   =  - ==-=\`.
                        |/// /  =    `. - =   == - =.=  = |
                       ///    -   -    )  - -   =  - == - =/\
                      /'      =  - = `.  - - = --   =-  = - \
                     '         -    - `) - _-  =- =  - = -   `
                    /        -  -   - .-'     ////-\     =   -\
                   |            =-  /         ((' `))          |
                   \   -   -        |    .    | o o|     -     /
                    \       -      /    \_)    \_-/        - /
                     `,    -     ,'      '--..___,'    =    ,'
                      `-._   _,-'     @....@     `.  - _,-'
                         `""`         |    |        `""`
                                     /    /\
                                    /  . /  \
                                   /   V     \
                                  /           \
                                 /    ___      \
                                /   /o o \      \
                               |   |  >  |       |
                               |   | --- |       |
                               |    \___/        |
                                \     ||        /
                                 \    ||       /
                                  \   ||      /
                                   \__||___  /
                                  /         \
                                 /  JUDAS    \
                                /   /---\     \
                               /   | o o |     \
                              /    |  <  |      \
                             /     | --- |       \
                            /       \___/         \
                           /         | |           \
                          /     _____|_|_____       \
                         /     /             \       \
                        /     /   BRUTUS      \       \
                       /     |    /---\        |       \
                      /      |   | o o |       |        \
                     /       |   |  <  |       |         \
                    /        |   | --- |       |          \
                   /         |    \___/        |           \
                  /          |     | |         |            \
                 /           |_____|_|_________|  CASSIUS    \
                /                  | |              /---\     \
               /                   | |             | o o |     \
              /                 ___| |___          |  <  |      \
             /                 /  /   \  \         | --- |       \
            /_________________/__/_____\__\________\___/_________\
           ////////////////////////////////////////////////////////////////
          ////////////////////////////////////////////////////////////////
         ////////////////////////////////////////////////////////////////
    """
    
    print("Detailed Satan/Lucifer ASCII Art (Dante's Inferno):")
    print(satan_art)
    
    print("\n" + "="*70)
    print("For even more detail, use the image converter with a Doré illustration!")
    print("="*70)

def main():
    """Main function to demonstrate ASCII art conversion"""
    
    print("ASCII Art Converter - Multiple Styles Available\n")
    
    # Show demo
    demo_satan_ascii()
    
    print("\n\nTo convert an image to ASCII art:")
    print("1. Save an image (like the Doré Lucifer) to your computer")
    print("2. Run: python ascii_art_converter.py <image_path>")
    print("\nAvailable styles:")
    print("  --style standard   : Basic ASCII characters")
    print("  --style detailed   : More detailed character set")
    print("  --style blocks     : Unicode block characters")
    print("  --style extended   : Maximum detail with 70+ characters")
    print("  --style dore       : Edge-detection style like Doré engravings")
    print("  --width 120        : Set output width (default: 80)")
    
    # If image path provided, convert it
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        # Parse optional arguments
        style = 'detailed'
        width = 80
        
        for i, arg in enumerate(sys.argv):
            if arg == '--style' and i + 1 < len(sys.argv):
                style = sys.argv[i + 1]
            elif arg == '--width' and i + 1 < len(sys.argv):
                width = int(sys.argv[i + 1])
        
        if os.path.exists(image_path):
            print(f"\nConverting {image_path} to ASCII art...")
            converter = ASCIIArtConverter(char_set=style, width=width)
            
            if style == 'dore':
                ascii_art = converter.create_dore_style_ascii(image_path)
            else:
                ascii_art = converter.image_to_ascii(image_path)
            
            print(ascii_art)
            
            # Save to file
            output_path = Path(image_path).stem + '_ascii.txt'
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(ascii_art)
            print(f"\nSaved to: {output_path}")
        else:
            print(f"\nError: Image file not found: {image_path}")

if __name__ == '__main__':
    main()