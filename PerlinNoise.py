import numpy as np
import random
import datetime


class Perlin:
    def __init__(self, w, h):
        self.width = w
        self.height = h
        self.initialSetSeed = 123456 # it is just to show function better not use it
        random.seed(self.initialSetSeed)
        self.permutationTable = [random.randint(0, 255) for i in range(1024)]


    def _GetRandGradientVector(self, x, y): # gives gradient vector using prime numbers
        v = ((x * 1836311903) ^ (y * 2971215073) + 4807526976) & 1023
        v = self.permutationTable[v]&3
        vectors = {0 : [1, 0], 1 : [-1, 0], 2 : [0, 1], 3 : [0, -1]}
        return vectors[v]
    

    def _NewPermutationTable(self): # gives new permutation table for multiParameterNoise, so that each noise could get new table
        self.permutationTable = [random.randint(0, 255) for i in range(1024)]
        

    @staticmethod
    def _GetSeed(): # generates seed using time (it was the most convinient)
        t = datetime.datetime.now()
        return int(t.strftime("%Y%m%d%H%M%S%f"))


    @staticmethod
    def _Lerp(a, b, t): # for linear interpolation using quintic parameter "t"
        return a * (1 - t) + b * t


    @staticmethod
    def _DotProd(a, b): # to calculate dot product of vectors
        return a[0] * b[0] + a[1] * b[1]


    @staticmethod
    def _QuinticCurve(t): # changing "t" along Quintic Curve 
        return t * t * t * (t * (t * 6 - 15) + 10)
    
    
    def Noise(self, x, y):
        # coordinates of top left of square
        left = int(x / self.width)
        top  = int(y / self.height)
        # local coordinates inside square
        localCordinateX = x/self.width - left
        localCordinateY = y/self.height - top
        # getting gradient vectors for all vertice of square
        topLeftGradient     = self._GetRandGradientVector(left,   top)
        topRightGradient    = self._GetRandGradientVector(left+1, top)
        bottomLeftGradient  = self._GetRandGradientVector(left,   top+1)
        bottomRightGradient = self._GetRandGradientVector(left+1, top+1)
        # vectors from vertice to point inside square 
        distanceToTopLeft     = [localCordinateX,   localCordinateY]
        distanceToTopRight    = [localCordinateX-1, localCordinateY]
        distanceToBottomLeft  = [localCordinateX,   localCordinateY-1]
        distanceToBottomRight = [localCordinateX-1, localCordinateY-1]
        
        # calculating dot product to interpolate between

        #       tx1--tx2
        #        |    |
        #       bx1--bx2   

        tx1 = self._DotProd(distanceToTopLeft,     topLeftGradient)
        tx2 = self._DotProd(distanceToTopRight,    topRightGradient)
        bx1 = self._DotProd(distanceToBottomLeft,  bottomLeftGradient)
        bx2 = self._DotProd(distanceToBottomRight, bottomRightGradient)

        # making interpolation parameters non-linear
        localCordinateX = self._QuinticCurve(localCordinateX)
        localCordinateY = self._QuinticCurve(localCordinateY)

        # interpolation
        tx = self._Lerp(tx1, tx2, localCordinateX)
        bx = self._Lerp(bx1, bx2, localCordinateX)
        tb = self._Lerp(tx,  bx,  localCordinateY) # actual result of interpolation

        return tb
    

    def multioctaveNoise(self, x, y, octaves, persistence=0.5, amplitude = 1):
        max, result = 0, 0

        while octaves > 0:
            max += amplitude
            result += self.Noise(x, y) * amplitude
            amplitude *= persistence
            x *= 2
            y *= 2
            octaves -= 1

        return result / max
    

    def multiParameterNoise(self, parameters):
        noise_values = np.zeros((self.height, self.width))

        for config in parameters:
            if "octaves" not in config:
                config["octaves"] = 6
            if "persistence" not in config:
                config["persistence"] = 0.5
            if "amplitude" not in config:
                config["amplitude"] = 1
            if "seed" not in config:
                config["seed"] = self._GetSeed()

            random.seed(config["seed"])
            octaves     = config["octaves"]
            amplitude   = config["persistence"]
            persistence = config["amplitude"]

            for i in range(self.height):
                for j in range(self.width):
                    noise_values[i][j] += (self.multioctaveNoise(j, i, octaves, persistence, amplitude))

            self._NewPermutationTable() # each new noise will get new table
        return noise_values
