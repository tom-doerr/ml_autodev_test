import sys
from pathlib import Path

# Add project root to Python path
root_dir = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(root_dir))
