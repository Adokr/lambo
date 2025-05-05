"""
Short demo on using LAMBO
"""
from lambo.segmenter.lambo import Lambo
import pathlib

if __name__ == '__main__':
    
    # Load the recommended model for Polish
    lambo = Lambo.get('English')
    
    # Provide text, including pauses (``(yy)``), emojis and turn markers (``<turn>``).
    text = "Simple sentences can't be enough... Some of us just ❤️ emojis. They should be tokens even when (yy) containing many characters, such as 👍🏿."
    
    # Perform segmentation
    document = lambo.segment(text)
    
    # Display the results
    document.print()
