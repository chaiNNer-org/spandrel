import os
import sys

# I fucking hate python. This hack is necessary to make our module import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
