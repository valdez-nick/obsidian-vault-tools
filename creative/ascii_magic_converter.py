#!/usr/bin/env python3
"""
ASCII Magic Converter - High Quality Image to ASCII Art
Uses the ascii-magic library for superior results
"""

import os
import sys

# Install ascii-magic if not present
try:
    import ascii_magic
except ImportError:
    print("Installing ascii-magic...")
    os.system(f"{sys.executable} -m pip install ascii-magic")
    import ascii_magic

def convert_to_ascii(image_path, columns=120, char_set=None, to_terminal=True):
    """
    Convert image to ASCII art using ascii-magic
    
    Args:
        image_path: Path to input image
        columns: Width in characters (default 120)
        char_set: Custom character set (optional)
        to_terminal: Print to terminal (True) or return string (False)
    """
    # Open image with PIL
    from PIL import Image
    img = Image.open(image_path)
    
    # Resize image to match desired columns
    aspect_ratio = img.height / img.width
    new_width = columns
    new_height = int(aspect_ratio * columns * 0.55)  # Adjust for character aspect ratio
    img = img.resize((new_width, new_height))
    
    # Convert to ASCII
    output = ascii_magic.from_pillow_image(img)
    
    if to_terminal:
        output.to_terminal()
    else:
        return output

def create_lucifer_ascii():
    """Create a detailed ASCII art of Lucifer/Satan"""
    
    # Extended ASCII art with shading
    lucifer = r"""
                                    .-.._
                              __  /`     '.
                           .-'  `/   (   o \
                          /|    /    `_     ;
                         | |   |      (_)   /
                         | |   |    _;-._ -'_
                         | `.  |  ./  / |/ \  `._
                          \  `.|_.;  /  ||  \    `-.
                           `-._;   /   ||   \      `.
                       ,-.     /  /    ||    \       \
                     ,'   `. /  /     ||     \  .-.  |
                    /       V  /  /|  ||  |\  \/   \ |
                   /   .-.    /  /_|  ||  |_\  |    ||
                  |   /   \  /  /__|  ||  |__\ |    ||
                  |  |  o  |/_-'   |  ||  |   `|    ||
                  |  |     |   .--{JUDAS}---.  |    ||
                  |  | --- |  /   _|  ||  |_   \|    ||
                  |   \   /  |   (_)  ||  (_)  ||    ||
                   \   `-'   |    |   ||   |   ||   /:|
                    \       /|    |   ||   |   ||  / :|
                     `.    / |   /    ||    \  || /  :|
                       `--'  |  |  BRUTUS   |  ||/   :|
                             |  |  .-""-.   |  |/    :|
                             |  | ( o  o )  |  /     :|
                             |  |  \  <  /  | /   CASSIUS
                             |  |   |--|    |/     .-.:|
                             |  |   |  |    /    .(o o)|
                             |   \  |  |   /      \\<//:|
                             |    `.|  |.-'        `-' :|
                          ___:_____/    \_____:________:|___
                         ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
                       ░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░
                      ░▒▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒░
                     ░▒▓███████████ COCYTUS █████████████▓▒░
                    ░▒▓████████████████████████████████████▓▒░
    """
    
    print("Detailed Lucifer ASCII Art (with shading):")
    print(lucifer)

def main():
    """Main function"""
    
    print("ASCII Magic Converter - Professional Quality ASCII Art\n")
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        # Parse optional arguments
        columns = 120
        for i, arg in enumerate(sys.argv):
            if arg == '--width' and i + 1 < len(sys.argv):
                columns = int(sys.argv[i + 1])
        
        if os.path.exists(image_path):
            print(f"Converting {image_path} to ASCII art...")
            print(f"Width: {columns} characters\n")
            
            try:
                # Convert and display
                convert_to_ascii(image_path, columns=columns)
                
                # Also save to file
                from PIL import Image
                img = Image.open(image_path)
                
                # Resize for columns
                aspect_ratio = img.height / img.width
                new_width = columns
                new_height = int(aspect_ratio * columns * 0.55)
                img = img.resize((new_width, new_height))
                
                output = ascii_magic.from_pillow_image(img)
                output_path = os.path.splitext(image_path)[0] + '_ascii_magic.txt'
                
                # Save using the to_file method
                output.to_file(output_path, monochrome=True)
                
                print(f"\n\nSaved to: {output_path}")
                
            except Exception as e:
                print(f"Error: {e}")
                print("\nTry a different image or check the file format.")
        else:
            print(f"Error: Image file not found: {image_path}")
    else:
        # Show demo
        create_lucifer_ascii()
        
        print("\n\nUsage:")
        print("  python ascii_magic_converter.py <image_path> [--width 120]")
        print("\nExample:")
        print("  python ascii_magic_converter.py dore_lucifer.jpg --width 150")
        print("\nThis will create a highly detailed ASCII art version!")

if __name__ == '__main__':
    main()