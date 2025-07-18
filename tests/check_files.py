import os
import matplotlib.pyplot as plt
import numpy as np

# Create a simple test plot
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 2])
ax.set_title('Test')

# Save to the figures directory
test_path = "../figures/test_check.png"
print(f"Saving to: {test_path}")
print(f"Absolute path: {os.path.abspath(test_path)}")

plt.savefig(test_path)
plt.close()

# Check if file exists
if os.path.exists(test_path):
    print(f"✓ File exists at relative path: {test_path}")
else:
    print(f"✗ File not found at relative path: {test_path}")

# Check absolute path
abs_path = os.path.abspath(test_path)
if os.path.exists(abs_path):
    print(f"✓ File exists at absolute path: {abs_path}")
else:
    print(f"✗ File not found at absolute path: {abs_path}")

# List all files in figures directory
figures_dir = "../figures"
print(f"\nAll files in {figures_dir}:")
if os.path.exists(figures_dir):
    for file in os.listdir(figures_dir):
        file_path = os.path.join(figures_dir, file)
        if os.path.isfile(file_path):
            print(f"  {file} ({os.path.getsize(file_path)} bytes)")
else:
    print("Figures directory does not exist!")

# Check current working directory
print(f"\nCurrent working directory: {os.getcwd()}") 