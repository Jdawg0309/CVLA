"""
CVLA - Complete Visual Linear Algebra
Main entry point for the enhanced 3D linear algebra visualizer
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.app import App

def main():
    """Main entry point for CVLA application."""
    try:
        app = App()
        app.run()
    except Exception as e:
        print(f"Error starting CVLA: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()