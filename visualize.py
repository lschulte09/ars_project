import numpy as np

# Path to your .npy file
filename = 'robot_map.npy'

# Load it – if it contains objects (pickled Python objects), set allow_pickle=True
data = np.load(filename, allow_pickle=False)

# Now `data` is a NumPy array. You can inspect it:
print("Shape:", data.shape)
print("Dtype:", data.dtype)
print("Contents (first 5 elements):", data.flat[:5])

# Example: if it's a 2D occupancy grid, you can visualize it via pygame or matplotlib:
import matplotlib.pyplot as plt
plt.imshow(data, cmap='gray')
plt.title('Loaded .npy array')
plt.colorbar()
plt.show()
