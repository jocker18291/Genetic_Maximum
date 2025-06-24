import numpy as np
import random
from deap import base, creator, tools

def func(x):
    return (pow(np.exp(), x) * np.sin(np.pi * x) + 1) / x

# range is (0.500, 2.500) so no if x = 0 needed

BOUNDS = (0.500, 2.500) # range where the func(x) will be optimized
POP_SIZE = 100 # population count
GENS = 50 # no. of generations
MUTPB = 0.01 # mutation probability
CXPB = 0.7 # crossing probability