import numba
from numba import cuda
import numpy as np
import random
import datetime



initialSetSeed = 123456 # it is just to show function do not use it
random.seed(initialSetSeed)
permutationTable = np.array([random.randint(0, 255) for i in numba.prange(1024)], dtype=np.int32)



def GetRandGradientVector(x, y): # gives gradient vector using prime numbers
    v = ((x * 1836311903) ^ (y * 2971215073) + 4807526976) & 1023
    v = permutationTable[v]&3
    vectors = np.array([
        np.array([1, 0], dtype=np.int32),
        np.array([0, 1], dtype=np.int32),
        np.array([-1, 0], dtype=np.int32),
        np.array([0, -1], dtype=np.int32),
        ], 
        dtype=np.int32)
    return vectors[v]




def NewPermutationTable(): # gives new permutation table for multiParameterNoise, so that each noise could get new table
    global permutationTable
    permutationTable = np.array([random.randint(0, 255) for i in numba.prange(1024)], dtype=np.int32)



def GetSeed(): # generates seed using time (it was the most convinient)
    t = datetime.datetime.now()
    return int(t.strftime("%Y%m%d%H%M%S%f"))



@cuda.jit(device=True)
def Lerp(a, b, t): # for linear interpolation using quintic parameter "t"
    return a * (1 - t) + b * t



@cuda.jit(device=True)
def DotProd(a, b): # to calculate dot product of vectors
    return a[0] * b[0] + a[1] * b[1]



@cuda.jit(device=True)
def QuinticCurve(t): # changing "t" along Quintic Curve 
    return t * t * t * (t * (t * 6 - 15) + 10)



@numba.jit(fastmath=True, parallel=True)
def Noise(fx, fy):
    # coordinates of top left of square
    left = int(fx)
    top = int(fy)
    # local coordinates inside square
    pointInQuadX = fx - left
    pointInQuadY = fy - top
    # getting gradient vectors for all vertice of square
    topLeftGradient     = GetRandGradientVector(left, top)
    topRightGradient    = GetRandGradientVector(left+1, top)
    bottomLeftGradient  = GetRandGradientVector(left, top+1)
    bottomRightGradient = GetRandGradientVector(left+1, top+1)
    # vectors from vertice to point inside square 
    distanceToTopLeft     = np.array([pointInQuadX, pointInQuadY], dtype=np.float32)
    distanceToTopRight    = np.array([pointInQuadX-1, pointInQuadY], dtype=np.float32)
    distanceToBottomLeft  = np.array([pointInQuadX, pointInQuadY-1], dtype=np.float32)
    distanceToBottomRight = np.array([pointInQuadX-1, pointInQuadY-1], dtype=np.float32)
    
    # calculating dot product to interpolate between

    #       tx1--tx2
    #        |    |
    #       bx1--bx2   

    tx1 = DotProd(distanceToTopLeft, topLeftGradient)
    tx2 = DotProd(distanceToTopRight, topRightGradient)
    bx1 = DotProd(distanceToBottomLeft, bottomLeftGradient)
    bx2 = DotProd(distanceToBottomRight, bottomRightGradient)

    # making interpolation parameters non-linear
    pointInQuadX = QuinticCurve(pointInQuadX)
    pointInQuadY = QuinticCurve(pointInQuadY)

    # interpolation
    tx = Lerp(tx1, tx2, pointInQuadX)
    bx = Lerp(bx1, bx2, pointInQuadX)
    tb = Lerp(tx, bx, pointInQuadY) # actual result of interpolation
    return tb



@numba.jit(fastmath=True, parallel=True)
def multioctaveNoise(fx, fy, octaves, persistence=0.5, amplitude = 1):
    max, result = 0, 0

    for i in numba.prange(octaves):
        max += amplitude
        result += Noise(fx, fy) * amplitude
        amplitude *= persistence
        fx *= 2
        fy *= 2

    return result / max



def multiParameterNoise(size_x, size_y, parameters):
    noise_values = np.zeros((size_y, size_x))

    for config in parameters:
        if "octaves" not in config:
            config["octaves"] = 6
        if "persistence" not in config:
            config["persistence"] = 0.5
        if "amplitude" not in config:
            config["amplitude"] = 1
        if "seed" not in config:
            config["seed"] = GetSeed()

        random.seed(config["seed"])
        octaves     = config["octaves"]
        amplitude   = config["persistence"]
        persistence = config["amplitude"]

        for i in range(size_y):
            for j in range(size_x):
                noise_values[i][j] += (multioctaveNoise(i/size_y, j/size_x, octaves, persistence, amplitude))

        NewPermutationTable() # each new noise can get new table
    return noise_values





@numba.jit(fastmath=True, parallel=True)
def main():
    size_x, size_y = 500, 500

    # parameters = [
    #     {"octaves" : 24, "persistence" : 0.9, "amplitude" : 1,},
    #     {"octaves" : 18, "persistence" : 0.7, "amplitude" : 1,},
    #     {"octaves" : 12, "persistence" : 0.5, "amplitude" : 1,},
    #     {"octaves" : 10, "persistence" : 0.4, "amplitude" : 1,},
    #     {"octaves" :  6, "persistence" : 0.3, "amplitude" : 1,},
    # ]
    # noise_values = multiParameterNoise(size_x, size_y, parameters)
    # 00:15 for 100x100 02:39 for 500x500   30:29 for 500x500 multiparameter noise

    noise_values = np.zeros((size_y, size_x))
    for i in range(size_y):
        for j in range(size_x):
            noise_values[i][j] = multioctaveNoise(i/size_y, j/size_x, 6)
    return noise_values




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    noise_values = main()
    # Plot the noise values
    plt.imshow(noise_values, cmap='gray', origin="lower")
    plt.show()