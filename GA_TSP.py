#%matplotlib notebook
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from random import sample, shuffle
from sklearn.metrics.pairwise import pairwise_distances


########################### CREATE INPUT ################################

data = [        [10,10], [15,30], [45,77],  [20,50],  [60,50],
                [60,15], [90,25], [90,79],  [54,18],  [21,10],
                [63,91], [16,71], [26,29],  [58,39],  [33,15],
                [33,55], [80,90], [100,79], [100,59], [22,80],
                [33,81], [48,88], [50,77],  [90,76],  [20,56],
                [80,60], [30,5],  [20,10],  [0,30],   [100,80],
                [11,54], [11,90], [11,18],  [80,43],  [43,81],
                [76,67], [57,50], [99,21],  [99,88],  [5,31]
        ]

# FUNCTION THAT CREATES RANDOM SEQUENCE
def random_sequence():
    ran_seq = []
    r = 0

    while len(ran_seq) < 40:  #creating a list of integers <39 in random order
        r = random.randint(1,40)
        if r not in ran_seq:
            ran_seq.append(r)

    return ran_seq


# FUNCTION THAT USES RANDOM SEQUENCE TO BUILD A COORDINATE LIST
def coords_indx_res():

    coords_indx = []
    count1 = 0
    count2 = 0

    restriction_1 = [5,10,15,20,25,30,35,40]     # only visit 3 of these
    restriction_2 = [2,4,6,8,10,20,22,32,33,35]  # only visit 5 of these

    ran_seq = random_sequence()

    for i in ran_seq: #setting up restrictions

       #if a coordinate is in both restrictions
        if i in restriction_1 and i in restriction_2:
            if count1 != 3 and count2 != 5:
                coords_indx.append(i)
                count1 += 1
                count2 += 1

        #if a coordinate is in restriction_1
        elif i in restriction_1:
            if count1 != 3:
                coords_indx.append(i)
                count1 += 1

        #if a coordinate is in restriction_2
        elif i in restriction_2:
            if count2 != 5:
                coords_indx.append(i)
                count2 += 1

        else:
            coords_indx.append(i)

    return(ran_seq, coords_indx, len(coords_indx))


# OUTPUTTING INDICIES OF THE CHOSEN COORDINATES
ran_seq, coords_indx, lc = coords_indx_res()
print(ran_seq, "\nLength is: ", len(ran_seq))
print(coords_indx, "\nLength is: ", len(coords_indx))


# LOOK AT THE INDICIES IN THE coords_indx, AND EXTRACT CORRESPONDING COORDINATES
def extract(coords_list_in, coords_indx):

    # first add the depot coordinate
    coords_list = [(0,0)]
    coords_indx_sort = sorted(coords_indx)

    # outputing coordinates
    for i in coords_indx_sort:
        coords_list.append(coords_list_in[i-1])

    #finish at the depot coordinate
    coords_list.append([0,0])

    return coords_list, coords_indx_sort

# OUTPUTTING COORDINATES TO coords_list
coords_list, sorty = extract(data, coords_indx)



############################# GENETIC ALGORITHM #############################

# CREATE INPUT DATA: GENES, CHROMOSOMES AND DISTANCES
n_genes = len(coords_list)
n_chromosomes = 300

customers = coords_list
distances = pairwise_distances(customers, metric='euclidean')


# GENERATE POPULATION OF CHROMOSOMES
def getPopulation(n_genes, n_chromosomes):  #300 chromosomes of 31-33 genes
    population = []
    n = 0
    while n < n_chromosomes:
        chromosome = np.random.permutation(n_genes)
        if not any([np.array_equal(chromosome, x) for x in population]):
            population.append(chromosome)
            n += 1
    return population


# OUTPUTTING THE INITIAL POPULATION
population = getPopulation(n_genes, n_chromosomes)


# FITNESS FUNCTION: SUM DISTANCES BETWEEN CUSTOMERS
def fitness(chromosome):
    fitness = 0
    n = len(chromosome)
    global distances

    #for i in range(len(chromosome) - 1):
    #    fitness += distances[chromosome[i]][chromosome[i+1]]

    xs = [customers[chromosome[i % n]][0] for i in range(n+1)]
    ys = [customers[chromosome[i % n]][1] for i in range(n+1)]

    for k in range(len(xs)-1):
        fitness += math.sqrt((xs[k] - xs[k+1])**2 + (ys[k] - ys[k+1])**2)


    return fitness


# SELECTION METHODS, DEF: STOCHASTIC UNIVERSAL SAMPLING
def probabilisticSelection(population, selection_factor, p=[1, 0]):
    total_fitness = sum([fitness(x) for x in population])
    population = [[x, (p[0]*(total_fitness - fitness(x))+p[1])/(p[0]*(total_fitness*(len(population)-1))+p[1])] for x in population]
    population.sort(key=lambda x: x[1])
    population = [[x[0], y] for x, y in zip(population, np.cumsum([x[1] for x in population]))]

    new_population = []

    r = np.random.rand()
    for i in range(selection_factor):
        selected_chromosome = [x for x in population if x[1] >= (r + i/selection_factor)%1][0]
        new_population.append(selected_chromosome[0])

    return new_population


# CROSSOVER
def crossover(parent1, parent2):
    n_genes = len(parent1)
    cross_point1, cross_point2 = sample(range(n_genes), 2)
    if (cross_point1 > cross_point2):
        tmp = cross_point1
        cross_point1 = cross_point2
        cross_point2 = tmp

    child1 = np.array(parent1)
    child2 = np.array(parent2)
    j1 = cross_point1
    j2 = cross_point2
    for i in range(cross_point1, cross_point2):
        while parent2[j1] not in parent1[cross_point1:cross_point2]:
            j1 = (j1+1)%n_genes
        child1[i] = parent2[j1]
        j1 = (j1+1)%n_genes

        while parent1[j2] not in parent2[cross_point1:cross_point2]:
            j2 = (j2+1)%n_genes
        child2[i] = parent1[j2]
        j2 = (j2+1)%n_genes

    return child1, child2#np.array(child1), np.array(child2)


# MUTATION
def mutation(chromosome):
    mutated = np.array(chromosome)
    gene1, gene2 = sample(range(len(chromosome)), 2)
    mutated[gene1] = chromosome[gene2]
    mutated[gene2] = chromosome[gene1]
    return mutated


# APPLY CROSSOVER AND MUTATION TO THE PARENTS
def createOffspring(parents, p=0.1):
    offspring = []

    for i in range(len(parents)):
        p1, p2 = sample(parents, 2)
        c1, c2 = crossover(p1, p2)
        if (not any([np.array_equal(c1, x) for x in parents]) and
            not any([np.array_equal(c1, x) for x in offspring])):
            offspring.append(c1)
        if (not any([np.array_equal(c2, x) for x in parents]) and
            not any([np.array_equal(c2, x) for x in offspring])):
            offspring.append(c2)

    for x in parents:
        if np.random.rand() <= p:
            c = mutation(x)
            if (not any([np.array_equal(c, x) for x in parents]) and
                not any([np.array_equal(c, x) for x in offspring])):
                offspring.append(c)

    return offspring


# ELITISM REPLACEMENT
def elitismReplacement(population, offspring, n_elite):

    population.sort(key=lambda x: fitness(x))
    new_population = population[:n_elite]
    offspring.sort(key=lambda x: fitness(x))
    new_population.extend(offspring[:(len(population) - n_elite)])

    return new_population


# GENETIC ALGORITHM
def TSP_GA(runs):
    global population

    for i in range(runs):
        parents = probabilisticSelection(population, 150)
        offspring = createOffspring(parents)
        population = elitismReplacement(population, offspring, 100)

        best = min(population, key=lambda x: fitness(x))
        plotdata = [customers[x] for x in best]
    route = []


    # SETTING UP PARAMETERS TO OUTPUT RESULT
    for i in best:
        route.append(i)

    n = len(coords_list)
    xs = [coords_list[route[i % n]][0] for i in range(n+1)]
    ys = [coords_list[route[i % n]][1] for i in range(n+1)]

    total_dist = 0
    for k in range(len(xs)-1):
            total_dist += math.sqrt((xs[k] - xs[k+1])**2 + (ys[k] - ys[k+1])**2)

    plt.plot(xs, ys, 'ob-')
    plt.show()

    print("Total Distance: ", total_dist)
    print(customers)



TSP_GA(250)
