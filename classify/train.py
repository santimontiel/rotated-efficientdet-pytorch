# Stdlib imports.
import sys
from pathlib import Path

# Configure relative path.
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if ROOT not in sys.path:
    sys.path.append(str(ROOT))

# Import EfficientNet from custom package.
from red.efficientnet import EfficientNet

model = EfficientNet("b0", 3, 3)
print(model)