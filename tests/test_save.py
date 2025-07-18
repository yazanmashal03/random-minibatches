import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np

# Check matplotlib backend
print(f"Matplotlib backend: {matplotlib.get_backend()}")

# Test basic plot saving
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
ax.set_title('Test Plot')

# Try saving to different locations
test_paths = [
    "test_plot.png",
    "../figures/test_plot.png", 
    "/Users/yazanmashal/Desktop/University/random_minibatches/figures/test_plot.png"
]

for path in test_paths:
    try:
        print(f"Trying to save to: {path}")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if os.path.exists(path):
            print(f"✓ Successfully saved to {path}")
            print(f"  File size: {os.path.getsize(path)} bytes")
        else:
            print(f"✗ File not found at {path}")
            
    except Exception as e:
        print(f"✗ Error saving to {path}: {e}")

print("\nCurrent working directory:", os.getcwd())
print("Figures directory exists:", os.path.exists("../figures"))
print("Figures directory absolute path:", os.path.abspath("../figures")) 