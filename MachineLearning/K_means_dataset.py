import numpy as np

def generate_double_moon(n_samples, d, r, w):
    """
    Generates a double moon dataset.
    
    Parameters:
    n_samples (int): Total number of samples to generate.
    distance (float): Vertical distance between the two moons.
    radius (float): Radius of each moon.
    width (float): Thickness of each moon.
    
    Returns:
    X (ndarray): Array of shape (n_samples, 2) containing the data points.
    y (ndarray): Array of shape (n_samples,) containing the labels.
    """
    n_samples_per_moon = n_samples // 2
    
    # Generate the upper moon (Region A)
    theta = np.random.uniform(0, np.pi, n_samples_per_moon)
    radius = np.random.uniform(r - w/2, r + w/2, n_samples_per_moon)
    x_a = radius * np.cos(theta)
    y_a = radius * np.sin(theta) 
    
    # Generate the lower moon (Region B)
    x_b = radius * np.cos(theta) + r
    y_b = -radius * np.sin(theta) - d
    
    # Combine the moons
    X = np.vstack((np.column_stack((x_a, y_a)), np.column_stack((x_b, y_b))))
    y = np.hstack((np.zeros(n_samples_per_moon), np.ones(n_samples_per_moon)))
    
    return X, y

# Example usage
n_samples = 1000  # Total number of samples
distance = 1      # Vertical distance between the moons
radius = 10       # Radius of the moons
width = 4         # Width/thickness of the moons

for distance in range(-6, 2):
    filename = f'data_{distance}.txt'
    X, y = generate_double_moon(n_samples, distance, radius, width)
    np.savetxt(filename, X, delimiter=' ')


# Plotting the dataset
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], s=5, color='red', label='Region A')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], s=5, color='blue', label='Region B')
plt.title('Double Moon Dataset')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.legend()
plt.show()
