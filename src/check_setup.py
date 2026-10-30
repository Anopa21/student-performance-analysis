"""Phase 1 setup check: confirms libraries are installed and the dataset file exists."""

import sys
from pathlib import Path

import matplotlib
import numpy
import pandas
import scipy
import seaborn
import sklearn

print("Python:       ", sys.version.split()[0])
print("pandas:       ", pandas.__version__)
print("numpy:        ", numpy.__version__)
print("matplotlib:   ", matplotlib.__version__)
print("seaborn:      ", seaborn.__version__)
print("scikit-learn: ", sklearn.__version__)
print("scipy:        ", scipy.__version__)

# Build the path relative to this file, so the script works
# no matter which folder you run it from.
project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "student-por.csv"

print()
if data_path.exists():
    size_kb = data_path.stat().st_size / 1024
    print(f"Dataset found: {data_path}")
    print(f"File size: {size_kb:.1f} KB")
else:
    print(f"Dataset NOT found at: {data_path}")
    print("Check that student-por.csv is inside the data/ folder.")