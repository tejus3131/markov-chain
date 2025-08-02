#!/usr/bin/env python3
"""
Launcher script for the Markov Chain Text Predictor Web GUI
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the web GUI
from chain.gui import main

if __name__ == "__main__":
    main()
