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


size_x, size_y = 500, 500                                       # if it calculates too long try 100, 100
noise = Perlin(size_x, size_y)


noise_example_1 = noise.multiParameterNoise(parameters)

noise_example_2 = np.zeros((size_y, size_x))
for i in range(size_y):
    for j in range(size_x):
        noise_example_2[i][j] = noise.multioctaveNoise(i, j, 6)

noise_example_3 = np.zeros((size_y, size_x))
for i in range(size_y):
    for j in range(size_x):
        noise_example_3[i][j] = noise.Noise(i, j)               # <-only one octave


# Plot the noise values
plt.imshow(noise_example_1, cmap='gray', origin = "lower")
plt.show()
plt.imshow(noise_example_2, cmap='gray', origin = "lower")
plt.show()
plt.imshow(noise_example_3, cmap='gray', origin = "lower")
plt.show()