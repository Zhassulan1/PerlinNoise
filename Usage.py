import matplotlib.pyplot as plt
import numpy as np
from PerlinNoise import Perlin




parameters = [
    {"octaves" : 24, "persistence" : 0.9, "amplitude" : 1,},
    {"octaves" : 18, "persistence" : 0.7, "amplitude" : 1,},
    {"octaves" : 12, "persistence" : 0.5, "amplitude" : 1,},
    {"octaves" : 10, "persistence" : 0.4, "amplitude" : 1,},
    {"octaves" :  6, "persistence" : 0.3, "amplitude" : 1,},
]

size_x, size_y = 500, 500                                               # if it calculates too long try 100, 100

noise = Perlin(size_x, size_y)                                          # width and height are passed to Perlin class

# The most simple way to use:
noise_example_1 = noise.multiParameterNoise(parameters)                 # multiple noises with different octaves
# multiParameterNoise method is the best option to generate levels and maps in simple games like snake, pacman, etc.
# Other methods like multioctaveNoise and Noise are not suitable for this purpose because noise will be too smooth.


noise_example_2 = np.zeros((size_y, size_x))                            # creating an empty array
for i in range(size_y):
    for j in range(size_x):
        noise_example_2[i][j] = noise.multioctaveNoise(i, j, octaves=6) # multiple octaves



# Plot the noise values into image
plt.imshow(noise_example_1, cmap='gray', origin = "lower")
plt.show()
plt.imshow(noise_example_2, cmap='gray', origin = "lower")
plt.show()