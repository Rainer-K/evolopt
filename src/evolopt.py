
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

minX, maxX = -1., 1.
minY, maxY = -1., 1.

xLim = np.array([minX, maxX])
yLim = np.array([minY, maxY])

minXY = np.array([minX, minY])
maxXY = np.array([maxX, maxY])


class Individual:

    def __init__(self, xy=None):
        if xy is None:
            self.xy = np.zeros(2)
        else:
            self.xy = xy

    def x(self):
        return self.xy[0]

    def y(self):
        return self.xy[1]


class Population:

    def __init__(self, size=100):
        self.individuals = [Individual() for i in range(size)]

    def randomize(self):
        for individual in self.individuals:
            for i in range(len(individual.xy)):
                individual.xy[i] = np.random.uniform(minXY[i], maxXY[i], 1)

    def add(self, individual):
        self.individuals.append(individual)

class Fitness:

    def __init__(self):
        self.amplitude = 1
        self.means = np.array([0.5, 0.5])
        self.variances = np.array([0.1, 0.1])

    def evaluate(self, individual):
        return self.function(individual.xy)

    def evaluatePopulation(self, population):
        return np.array([self.evaluate(individual) for individual in population])

    def function(self, xy):
        #               (   ( (x-x0)^2   (y-y0)^2 ) )
        # f(x,y) = A exp( - ( -------- + -------- ) )
        #               (   (  2 vx^2    2 vy^2   ) )
        return self.amplitude * np.exp(- np.sum((xy - self.means) ** 2 /
                                                (2 * self.variances)))


class RastriginFitness(Fitness):

    def function(self, xy):
        n = 2
        return np.abs(- self.amplitude * n \
               + np.sum(xy**2 - self.amplitude * np.cos(2 * np.pi * xy)))



class Plot:

    def __init__(self):
        pass

    def plot(self, fitness=None, population=None):

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        if not fitness is None:
            X = np.linspace(minX, maxX, 32, endpoint=True)
            Y = np.linspace(minY, maxY, 32, endpoint=True)
            X, Y = np.meshgrid(X, Y)

            XY = np.dstack((X, Y))
            Z = np.apply_along_axis(fitness.function, 2, XY)

            # Plot the surface.
            ax.plot_wireframe(X, Y, Z, color='g')

        if not population is None:
            X = np.array([ind.x() for ind in population.individuals])
            Y = np.array([ind.y() for ind in population.individuals])
            Z = np.apply_along_axis(fitness.function, 2, np.dstack((X,Y)))
            ax.scatter(X, Y, Z, c='b', s=100)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Fitness')

        plt.xlim(xLim)
        plt.ylim(yLim)
        plt.show()



fitness = RastriginFitness()

population = Population(size=10)
population.randomize()

plot = Plot()
plot.plot(fitness, population)

def optimize(population, fitness):

    def mutate(parent):
        child = Individual()
        for i in range(len(child.xy)):
            child.xy[i] = parent.xy[i] \
                          + np.random.uniform(0.25 * minXY[i], 0.25 * maxXY[i], 1)
        return child

    def recombine(parent1, parent2):
        child = Individual()
        for i in range(len(child.xy)):
            factor = np.random.uniform(0., 1., 1)
            child.xy[i] = factor * parent1.xy[i] + (1 - factor) * parent2.xy[i]
        return child

    numGenerations = 15
    populationSize = len(population.individuals)

    for generation in range(numGenerations):

        newPopulation = Population(0)

        # 100% mutation + 100% recombination

        for i in range(populationSize):
            parent = random.choice(population.individuals)
            child = mutate(parent)
            newPopulation.add(child)

        for i in range(populationSize):
            parent1 = random.choice(population.individuals)
            parent2 = random.choice(population.individuals)
            child = recombine(parent1, parent2)
            newPopulation.add(child)

        # sort (ascending)
        newPopulation.individuals.sort(key=lambda indiv: fitness.evaluate(indiv))

        # keep better half
        newPopulation.individuals = newPopulation.individuals[-populationSize:]

        # generation swap
        population = newPopulation

        plot.plot(fitness, population)


optimize(population, fitness)








