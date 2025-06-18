#!/usr/bin/env python3
"""
Better ASCII Art Converter
Creates more recognizable ASCII art with proper character mapping
"""

import os
import sys
from PIL import Image, ImageEnhance, ImageOps
import numpy as np

class BetterASCIIConverter:
    """Creates traditional ASCII art with recognizable output"""
    
    # Character sets from light to dark
    CHAR_SETS = {
        'simple': ' .:-=+*#%@',
        'standard': ' .\'`^",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$',
        'blocks': ' ░▒▓█',
        'detailed': ' .-\':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@',
        'classic': '@%#*+=-:. ',  # Reversed for traditional look
        'matrix': ' .,:;i1tfLCG08@',
    }
    
    def __init__(self, char_set='standard', width=80, invert=False):
        """
        Initialize converter
        
        Args:
            char_set: Which character set to use
            width: Target width in characters
            invert: Whether to invert the brightness
        """
        self.chars = self.CHAR_SETS.get(char_set, self.CHAR_SETS['standard'])
        if invert:
            self.chars = self.chars[::-1]
        self.width = width
    
    def convert(self, image_path: str, enhance_contrast=True, brightness=1.0) -> str:
        """
        Convert image to ASCII art
        
        Args:
            image_path: Path to input image
            enhance_contrast: Whether to enhance contrast
            brightness: Brightness adjustment (1.0 = normal)
            
        Returns:
            ASCII art string
        """
        # Load image
        img = Image.open(image_path)
        
        # Convert to grayscale
        img = img.convert('L')
        
        # Enhance contrast if requested
        if enhance_contrast:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(2.0)
        
        # Adjust brightness
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness)
        
        # Calculate height to maintain aspect ratio
        aspect_ratio = img.height / img.width
        height = int(aspect_ratio * self.width * 0.55)  # 0.55 for character aspect
        
        # Resize image
        img = img.resize((self.width, height), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        pixels = np.array(img)
        
        # Create ASCII art
        ascii_art = []
        for row in pixels:
            ascii_row = ''
            for pixel in row:
                # Map pixel to character
                char_index = int((pixel / 255) * (len(self.chars) - 1))
                ascii_row += self.chars[char_index]
            ascii_art.append(ascii_row)
        
        return '\n'.join(ascii_art)

def create_ascii_art_samples():
    """Show different ASCII art styles"""
    
    print("ASCII ART STYLE SAMPLES")
    print("=" * 50)
    
    # Simple gradient
    print("\nGradient Test:")
    chars = ' .:-=+*#%@'
    for i in range(len(chars)):
        print(chars[i] * 20)
    
    # ASCII art examples
    print("\n\nSimple Shapes:")
    
    # Circle
    print("\nCircle:")
    circle = """
       .-""-.
      /      \\
     |        |
     |   @@   |
     |        |
      \\      /
       '-...-'
    """
    print(circle)
    
    # Face
    print("\nFace:")
    face = """
      .-------.
     /  o   o  \\
    |     >     |
    |   \\___/   |
     \\         /
      '-------'
    """
    print(face)

def main():
    """Main function"""
    
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        image_path = sys.argv[1]
        
        # Parse options
        width = 80
        style = 'standard'
        invert = False
        
        for i, arg in enumerate(sys.argv):
            if arg == '--width' and i + 1 < len(sys.argv):
                width = int(sys.argv[i + 1])
            elif arg == '--style' and i + 1 < len(sys.argv):
                style = sys.argv[i + 1]
            elif arg == '--invert':
                invert = True
        
        print(f"Converting {image_path} to ASCII art...")
        print(f"Style: {style}, Width: {width} chars\n")
        
        # Create converter
        converter = BetterASCIIConverter(char_set=style, width=width, invert=invert)
        
        # Try different settings
        if style == 'auto':
            # Try multiple styles and let user choose
            styles_to_try = ['simple', 'standard', 'classic', 'blocks']
            
            for s in styles_to_try:
                print(f"\n{'=' * 40}")
                print(f"Style: {s}")
                print('=' * 40)
                
                conv = BetterASCIIConverter(char_set=s, width=min(width, 60))
                art = conv.convert(image_path)
                
                # Show preview (first 10 lines)
                lines = art.split('\n')
                for line in lines[:10]:
                    print(line)
                if len(lines) > 10:
                    print("... (truncated)")
        else:
            # Convert with specified style
            ascii_art = converter.convert(image_path)
            print(ascii_art)
            
            # Save to file
            output_path = os.path.splitext(image_path)[0] + f'_ascii_{style}.txt'
            with open(output_path, 'w') as f:
                f.write(ascii_art)
            print(f"\nSaved to: {output_path}")
    else:
        # Show help and examples
        create_ascii_art_samples()
        
        print("\n\nUSAGE:")
        print("  python better_ascii_converter.py <image> [options]")
        print("\nOPTIONS:")
        print("  --width <num>    : Output width (default: 80)")
        print("  --style <name>   : Character style")
        print("                     simple, standard, blocks, detailed, classic, matrix")
        print("                     auto (try all styles)")
        print("  --invert         : Invert brightness")
        print("\nEXAMPLES:")
        print("  python better_ascii_converter.py image.jpg --style classic --width 100")
        print("  python better_ascii_converter.py image.jpg --style auto")
        print("  python better_ascii_converter.py image.jpg --style blocks --invert")

if __name__ == '__main__':
    main()