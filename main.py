import numpy as np
import random
from deap import base, creator, tools

def func(x):
    return (np.exp(x) * np.sin(np.pi * x) + 1) / x

# range is (0.500, 2.500) so no if x = 0 needed

BOUNDS = (0.500, 2.500) # range where the func(x) will be optimized
POP_SIZE = 100 # population count
GENS = 50 # no. of generations
MUTPB = 0.01 # mutation probability
CXPB = 0.7 # crossing probability

creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # fitness class
creator.create("Individual", list, fitness=creator.FitnessMax) # individual class

toolbox = base.Toolbox()

toolbox.register("attr_float", random.uniform, 0.5, 2.5) # registering random function
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 1) # register individual
toolbox.register("population", tools.initRepeat, list, toolbox.individual) # register population

def evaluate(individual):
    x = individual[0]
    return (np.exp(x) * np.sin(np.pi * x) + 1) / x,